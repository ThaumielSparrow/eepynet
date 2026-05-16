from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from eepynet.constants import IGNORE_LABEL
from eepynet.utils import load_json


@dataclass(frozen=True)
class ChunkIndex:
    record_id: str
    subject_id: str
    start_epoch: int
    num_epochs: int


class SleepEDFChunkDataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split_manifest: str | Path | dict[str, Any],
        split: str,
        epochs_per_chunk: int = 128,
        stride: int | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.epochs_per_chunk = int(epochs_per_chunk)
        self.stride = int(stride or epochs_per_chunk)
        self.split = split
        self.manifest = (
            split_manifest
            if isinstance(split_manifest, dict)
            else load_json(split_manifest)
        )
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
        self.index = self._build_index()

    def _load_meta(self, record_id: str) -> dict:
        return load_json(self.processed_dir / record_id / "meta.json")

    def _build_index(self) -> list[ChunkIndex]:
        if self.split not in self.manifest["splits"]:
            raise KeyError(f"Unknown split '{self.split}'")

        chunks: list[ChunkIndex] = []
        for record_id in self.manifest["splits"][self.split]["record_ids"]:
            meta = self._load_meta(record_id)
            n_epochs = int(meta["num_epochs"])
            if n_epochs <= 0:
                continue
            starts = list(range(0, n_epochs, self.stride))
            for start in starts:
                chunks.append(
                    ChunkIndex(
                        record_id=record_id,
                        subject_id=str(meta["subject_id"]),
                        start_epoch=start,
                        num_epochs=min(self.epochs_per_chunk, n_epochs - start),
                    )
                )
        return chunks

    def _load_record(self, record_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
        if record_id not in self._arrays:
            record_dir = self.processed_dir / record_id
            x = np.load(record_dir / "x.npy", mmap_mode="r")
            y = np.load(record_dir / "y.npy", mmap_mode="r")
            meta = load_json(record_dir / "meta.json")
            self._arrays[record_id] = (x, y, meta)
        return self._arrays[record_id]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.index[idx]
        x_record, y_record, meta = self._load_record(item.record_id)
        start = item.start_epoch
        end = start + item.num_epochs

        x_chunk = np.zeros(
            (x_record.shape[0], self.epochs_per_chunk, x_record.shape[2]),
            dtype=np.float32,
        )
        y_chunk = np.zeros((self.epochs_per_chunk,), dtype=np.int64)
        mask = np.zeros((self.epochs_per_chunk,), dtype=bool)

        x_slice = np.asarray(x_record[:, start:end, :], dtype=np.float32)
        y_slice = np.asarray(y_record[start:end], dtype=np.int64)
        valid = y_slice != IGNORE_LABEL

        x_chunk[:, : item.num_epochs, :] = x_slice
        y_chunk[: item.num_epochs] = np.where(valid, y_slice, 0)
        mask[: item.num_epochs] = valid

        return {
            "x": torch.from_numpy(x_chunk),
            "y": torch.from_numpy(y_chunk),
            "mask": torch.from_numpy(mask),
            "subject_id": item.subject_id,
            "record_id": item.record_id,
            "start_epoch": start,
            "num_epochs": item.num_epochs,
            "sample_rate": int(meta["sample_rate"]),
        }


def compute_class_counts(
    processed_dir: str | Path,
    split_manifest: str | Path | dict[str, Any],
    split: str = "train",
    num_classes: int = 5,
) -> np.ndarray:
    manifest = split_manifest if isinstance(split_manifest, dict) else load_json(split_manifest)
    counts = np.zeros(num_classes, dtype=np.int64)
    for record_id in manifest["splits"][split]["record_ids"]:
        y = np.load(Path(processed_dir) / record_id / "y.npy", mmap_mode="r")
        valid = np.asarray(y) >= 0
        counts += np.bincount(np.asarray(y)[valid], minlength=num_classes)[:num_classes]
    return counts
