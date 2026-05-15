from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_device(preference: str | None = "auto") -> torch.device:
    preference = (preference or "auto").lower()
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "training.device is 'cuda', but torch.cuda.is_available() is false. "
                "Install a CUDA PyTorch build with "
                "`uv sync --no-group torch-cpu --extra cu128` or use device=auto/cpu."
            )
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, cpu")
