from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from eepynet.config import load_config
from eepynet.constants import IGNORE_LABEL, STAGE_TO_ID, normalize_stage_label
from eepynet.data.records import EDFPair, discover_edf_pairs
from eepynet.utils import ensure_dir, save_json


def annotations_to_epoch_labels(
    onsets: Sequence[float],
    durations: Sequence[float],
    descriptions: Sequence[str],
    num_epochs: int,
    epoch_seconds: float = 30.0,
) -> np.ndarray:
    """Assign hypnogram annotations to fixed epochs by maximum overlap."""

    labels = np.full(num_epochs, IGNORE_LABEL, dtype=np.int64)
    overlaps = np.zeros(num_epochs, dtype=np.float32)

    for onset, duration, description in zip(onsets, durations, descriptions):
        duration = float(duration)
        if duration <= 0:
            continue
        label = normalize_stage_label(str(description))
        ann_start = float(onset)
        ann_end = ann_start + duration
        first_epoch = max(0, int(np.floor(ann_start / epoch_seconds)))
        last_epoch = min(num_epochs - 1, int(np.ceil(ann_end / epoch_seconds)) - 1)
        if last_epoch < first_epoch:
            continue

        for epoch_idx in range(first_epoch, last_epoch + 1):
            epoch_start = epoch_idx * epoch_seconds
            epoch_end = epoch_start + epoch_seconds
            overlap = max(0.0, min(epoch_end, ann_end) - max(epoch_start, ann_start))
            if overlap > overlaps[epoch_idx]:
                overlaps[epoch_idx] = overlap
                labels[epoch_idx] = IGNORE_LABEL if label is None else label

    return labels


def trim_excess_wake(
    x: np.ndarray,
    y: np.ndarray,
    keep_wake_epochs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if keep_wake_epochs < 0:
        raise ValueError("keep_wake_epochs must be non-negative")

    sleep_epochs = np.flatnonzero(y != STAGE_TO_ID["W"])
    if len(sleep_epochs) == 0:
        return x, y, {"trim_start_epoch": 0, "trim_end_epoch": int(len(y))}

    start = max(0, int(sleep_epochs[0]) - keep_wake_epochs)
    end = min(len(y), int(sleep_epochs[-1]) + keep_wake_epochs + 1)
    return x[:, start:end, :], y[start:end], {
        "trim_start_epoch": start,
        "trim_end_epoch": end,
    }


def normalize_record(x: np.ndarray, mode: str | None) -> np.ndarray:
    if mode in {None, "none"}:
        return x.astype(np.float32, copy=False)
    if mode != "per_record_channel_zscore":
        raise ValueError(f"Unsupported normalization mode: {mode}")

    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def load_and_preprocess_pair(
    pair: EDFPair,
    selected_channels: Sequence[str],
    sample_rate: int,
    epoch_seconds: int,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    trim_wake_minutes: float | None,
    normalize: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "EDF preprocessing requires mne. Install dependencies with `uv sync`."
        ) from exc

    raw = mne.io.read_raw_edf(
        pair.psg_path,
        preload=True,
        include=list(selected_channels),
        verbose="ERROR",
    )
    missing = [channel for channel in selected_channels if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{pair.record_id} is missing channels: {missing}")

    raw.pick(list(selected_channels))
    raw.filter(
        l_freq=bandpass_low_hz,
        h_freq=bandpass_high_hz,
        picks=list(selected_channels),
        verbose="ERROR",
    )
    if float(raw.info["sfreq"]) != float(sample_rate):
        raw.resample(sample_rate, npad="auto", verbose="ERROR")

    samples_per_epoch = int(sample_rate * epoch_seconds)
    data = raw.get_data(picks=list(selected_channels)).astype(np.float32)
    num_epochs = data.shape[1] // samples_per_epoch
    if num_epochs <= 0:
        raise ValueError(f"{pair.record_id} has no complete {epoch_seconds}s epochs")

    data = data[:, : num_epochs * samples_per_epoch]
    x = data.reshape(len(selected_channels), num_epochs, samples_per_epoch)

    annotations = mne.read_annotations(pair.hypnogram_path)
    y = annotations_to_epoch_labels(
        annotations.onset,
        annotations.duration,
        annotations.description,
        num_epochs=num_epochs,
        epoch_seconds=epoch_seconds,
    )

    valid = y != IGNORE_LABEL
    dropped_epochs = int((~valid).sum())
    x = x[:, valid, :]
    y = y[valid]

    trim_meta = {"trim_start_epoch": 0, "trim_end_epoch": int(len(y))}
    if trim_wake_minutes is not None:
        keep_wake_epochs = int(round((float(trim_wake_minutes) * 60.0) / epoch_seconds))
        x, y, trim_meta = trim_excess_wake(x, y, keep_wake_epochs)

    x = normalize_record(x, normalize)
    y = y.astype(np.int64, copy=False)

    meta: dict[str, object] = {
        "record_id": pair.record_id,
        "subject_id": pair.subject_id,
        "study": pair.study,
        "psg_path": pair.psg_path,
        "hypnogram_path": pair.hypnogram_path,
        "channels": list(selected_channels),
        "sample_rate": sample_rate,
        "epoch_seconds": epoch_seconds,
        "samples_per_epoch": samples_per_epoch,
        "num_epochs": int(len(y)),
        "dropped_epochs": dropped_epochs,
        "normalization": normalize,
        **trim_meta,
    }
    return x, y, meta


def save_processed_record(
    pair: EDFPair,
    processed_dir: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    meta: dict[str, object],
) -> None:
    record_dir = ensure_dir(Path(processed_dir) / pair.record_id)
    np.save(record_dir / "x.npy", x.astype(np.float32, copy=False))
    np.save(record_dir / "y.npy", y.astype(np.int64, copy=False))
    save_json(meta, record_dir / "meta.json")


def preprocess_dataset(
    config: dict,
    force: bool = False,
    limit: int | None = None,
) -> list[dict[str, object]]:
    paths = config["paths"]
    data_cfg = config["data"]
    pairs = discover_edf_pairs(paths["data_root"], data_cfg.get("include_studies"))
    if limit is not None:
        pairs = pairs[:limit]

    processed_dir = Path(paths["processed_dir"])
    ensure_dir(processed_dir)
    processed: list[dict[str, object]] = []

    for pair in tqdm(pairs, desc="Preprocessing Sleep-EDF records"):
        record_dir = processed_dir / pair.record_id
        meta_path = record_dir / "meta.json"
        if meta_path.exists() and not force:
            processed.append({"record_id": pair.record_id, "status": "cached"})
            continue

        x, y, meta = load_and_preprocess_pair(
            pair=pair,
            selected_channels=data_cfg["selected_channels"],
            sample_rate=int(data_cfg["sample_rate"]),
            epoch_seconds=int(data_cfg["epoch_seconds"]),
            bandpass_low_hz=float(data_cfg["bandpass_low_hz"]),
            bandpass_high_hz=float(data_cfg["bandpass_high_hz"]),
            trim_wake_minutes=data_cfg.get("trim_wake_minutes"),
            normalize=data_cfg.get("normalize"),
        )
        save_processed_record(pair, processed_dir, x, y, meta)
        processed.append({"record_id": pair.record_id, "status": "processed"})

    save_json({"records": processed}, processed_dir / "preprocess_manifest.json")
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF Expanded EDF files.")
    parser.add_argument("--config", default="configs/eepynet.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild existing processed records.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    preprocess_dataset(config, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
