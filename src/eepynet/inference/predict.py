from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from eepynet.config import load_config
from eepynet.constants import ID_TO_STAGE
from eepynet.models.eepynet import EepyNet, load_model_checkpoint
from eepynet.utils import ensure_dir, get_torch_device, load_json


@torch.no_grad()
def predict_record_logits(
    model: EepyNet,
    x: np.ndarray,
    chunk_size: int,
    stride: int,
    device: torch.device,
    use_amp: bool = False,
) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected x shape [C,E,T], got {x.shape}")
    if chunk_size <= 0 or stride <= 0:
        raise ValueError("chunk_size and stride must be positive")

    model.eval()
    channels, num_epochs, samples = x.shape
    num_classes = int(model.model_config["num_classes"])
    logits_sum = np.zeros((num_epochs, num_classes), dtype=np.float64)
    counts = np.zeros((num_epochs,), dtype=np.float64)

    for start in range(0, num_epochs, stride):
        end = min(num_epochs, start + chunk_size)
        valid_len = end - start
        chunk = np.zeros((1, channels, chunk_size, samples), dtype=np.float32)
        chunk[0, :, :valid_len, :] = np.asarray(x[:, start:end, :], dtype=np.float32)
        tensor = torch.from_numpy(chunk).to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(tensor).detach().cpu().numpy()[0, :valid_len, :]
        logits_sum[start:end] += logits
        counts[start:end] += 1.0

    counts = np.maximum(counts[:, None], 1.0)
    return (logits_sum / counts).astype(np.float32)


def maybe_median_smooth(labels: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return labels
    if kernel_size % 2 == 0:
        kernel_size += 1
    from scipy.signal import medfilt

    return medfilt(labels, kernel_size=kernel_size).astype(np.int64)


def save_hypnogram_csv(
    labels: np.ndarray,
    logits: np.ndarray,
    output_path: str | Path,
    epoch_seconds: int,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "start_second", "stage", "label_id", "logit_w", "logit_n1", "logit_n2", "logit_n3", "logit_rem"])
        for epoch_idx, label in enumerate(labels.tolist()):
            writer.writerow(
                [
                    epoch_idx,
                    epoch_idx * epoch_seconds,
                    ID_TO_STAGE[int(label)],
                    int(label),
                    *[float(value) for value in logits[epoch_idx].tolist()],
                ]
            )


def predict_processed_record(
    config: dict,
    checkpoint_path: str | Path,
    record_id: str,
    output_path: str | Path | None = None,
) -> Path:
    device = get_torch_device(config["training"].get("device", "auto"))
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    model, _ = load_model_checkpoint(checkpoint_path, map_location=device)
    model.to(device)

    record_dir = Path(config["paths"]["processed_dir"]) / record_id
    x = np.load(record_dir / "x.npy", mmap_mode="r")
    meta = load_json(record_dir / "meta.json")
    logits = predict_record_logits(
        model,
        x,
        chunk_size=int(config["inference"]["chunk_size"]),
        stride=int(config["inference"]["stride"]),
        device=device,
        use_amp=use_amp,
    )
    labels = logits.argmax(axis=-1).astype(np.int64)
    labels = maybe_median_smooth(labels, int(config["inference"].get("median_smoothing", 0)))

    if output_path is None:
        output_path = (
            Path(config["paths"]["output_dir"])
            / "predictions"
            / f"{record_id}_predicted_hypnogram.csv"
        )
    save_hypnogram_csv(labels, logits, output_path, int(meta["epoch_seconds"]))
    return Path(output_path)


def record_ids_from_split(config: dict, split: str) -> list[str]:
    manifest = load_json(config["paths"]["split_path"])
    return list(manifest["splits"][split]["record_ids"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full-night EepyNet inference.")
    parser.add_argument("--config", default="configs/eepynet.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--output", default=None, help="CSV path for a single --record-id run.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or Path(config["paths"]["checkpoint_dir"]) / "best.pt"

    if args.record_id is None and args.split is None:
        raise SystemExit("Provide either --record-id or --split.")
    if args.record_id is not None and args.split is not None:
        raise SystemExit("Use --record-id or --split, not both.")

    if args.record_id is not None:
        output = predict_processed_record(config, checkpoint, args.record_id, args.output)
        print(output)
        return

    for record_id in record_ids_from_split(config, args.split):
        output = predict_processed_record(config, checkpoint, record_id)
        print(output)


if __name__ == "__main__":
    main()
