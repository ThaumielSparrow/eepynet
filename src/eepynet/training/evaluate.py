from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eepynet.config import load_config
from eepynet.data.dataset import SleepEDFChunkDataset
from eepynet.models.eepynet import load_model_checkpoint
from eepynet.training.metrics import save_confusion_matrix_plot
from eepynet.training.trainer import build_dataloader, evaluate_model
from eepynet.utils import ensure_dir, get_torch_device, save_json


def evaluate_checkpoint(config: dict, checkpoint_path: str | Path, split: str) -> dict:
    device = get_torch_device(config["training"].get("device", "auto"))
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype_name = str(config["training"].get("amp_dtype", "bf16")).lower()
    amp_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}[amp_dtype_name]
    model, checkpoint = load_model_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    class_weights = torch.tensor(checkpoint.get("class_weights", []), dtype=torch.float32, device=device)
    if class_weights.numel() == 0:
        class_weights = None

    dataset = SleepEDFChunkDataset(
        processed_dir=config["paths"]["processed_dir"],
        split_manifest=config["paths"]["split_path"],
        split=split,
        epochs_per_chunk=int(config["dataset"]["epochs_per_chunk"]),
        stride=int(config["dataset"]["eval_stride"]),
    )
    loader = build_dataloader(config, dataset, shuffle=False)
    focal_gamma = float(config["training"].get("focal_gamma", 0.0))
    smooth_window = int(config["training"].get("eval_smooth_window", 1))
    return evaluate_model(
        model, loader, device,
        class_weights=class_weights,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        focal_gamma=focal_gamma,
        smooth_window=smooth_window,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an EepyNet PyTorch checkpoint.")
    parser.add_argument("--config", default="configs/eepynet.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or Path(config["paths"]["checkpoint_dir"]) / "best.pt"
    metrics = evaluate_checkpoint(config, checkpoint, args.split)
    output_dir = ensure_dir(config["paths"]["output_dir"])
    save_json(metrics, output_dir / f"{args.split}_metrics.json")
    save_confusion_matrix_plot(metrics["confusion_matrix"], output_dir / f"{args.split}_confusion_matrix.png")
    print(
        f"{args.split}: macro_f1={metrics['macro_f1']:.4f} "
        f"kappa={metrics['cohen_kappa']:.4f} accuracy={metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
