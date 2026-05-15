from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from eepynet.config import load_config
from eepynet.constants import CLASS_NAMES
from eepynet.data.dataset import SleepEDFChunkDataset, compute_class_counts
from eepynet.models.eepynet import EepyNet
from eepynet.training.losses import class_weights_from_counts, masked_weighted_cross_entropy
from eepynet.training.metrics import compute_metrics, save_confusion_matrix_plot
from eepynet.utils import ensure_dir, get_torch_device, load_json, save_json, seed_everything


def build_dataloader(
    config: dict[str, Any],
    split: str,
    shuffle: bool,
) -> DataLoader:
    dataset_cfg = config["dataset"]
    stride_key = "train_stride" if split == "train" else "eval_stride"
    dataset = SleepEDFChunkDataset(
        processed_dir=config["paths"]["processed_dir"],
        split_manifest=config["paths"]["split_path"],
        split=split,
        epochs_per_chunk=int(dataset_cfg["epochs_per_chunk"]),
        stride=int(dataset_cfg[stride_key]),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["x"].to(device, non_blocking=True),
        batch["y"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
    )


def train_one_epoch(
    model: EepyNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler, # pyright: ignore[reportPrivateImportUsage]
    device: torch.device,
    class_weights: torch.Tensor,
    use_amp: bool,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x, y, mask = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = masked_weighted_cross_entropy(logits, y, mask, class_weights)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_model(
    model: EepyNet,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for batch in loader:
        x, y, mask = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = masked_weighted_cross_entropy(logits, y, mask, class_weights)
        losses.append(float(loss.detach().cpu()))

        valid = (mask.bool() & (y >= 0)).detach().cpu().numpy()
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        y_true.append(labels[valid])
        y_pred.append(preds[valid])

    true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    metrics = compute_metrics(true, pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def build_scheduler(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    name = str(config["training"].get("scheduler", "cosine")).lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config["training"]["epochs"])),
        )
    if name in {"plateau", "reduce_on_plateau", "reduce-on-plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=5,
            factor=0.5,
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def save_checkpoint(
    path: str | Path,
    model: EepyNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    config: dict[str, Any],
    class_weights: torch.Tensor,
    metrics: dict[str, Any],
    best_score: float,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model.model_config,
            "config": config,
            "class_names": CLASS_NAMES,
            "class_weights": class_weights.detach().cpu().tolist(),
            "metrics": metrics,
            "best_score": best_score,
        },
        path,
    )


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config["training"]["seed"]))
    checkpoint_dir = ensure_dir(config["paths"]["checkpoint_dir"])
    output_dir = ensure_dir(config["paths"]["output_dir"])
    split_manifest = load_json(config["paths"]["split_path"])

    device = get_torch_device(config["training"].get("device", "auto"))
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)

    class_counts = compute_class_counts(
        config["paths"]["processed_dir"],
        split_manifest,
        split="train",
        num_classes=int(config["model"]["num_classes"]),
    )
    class_weights = class_weights_from_counts(class_counts).to(device)

    model = EepyNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) # pyright: ignore[reportPrivateImportUsage]

    monitor = str(config["training"].get("monitor", "macro_f1"))
    patience = int(config["training"].get("early_stopping_patience", 20))
    best_score = -float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            class_weights,
            use_amp,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            class_weights=class_weights,
            use_amp=use_amp,
        )
        val_metrics["train_loss"] = train_loss

        score = float(val_metrics[monitor])
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()

        is_best = score > best_score
        if is_best:
            best_score = score
            bad_epochs = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                class_weights,
                val_metrics,
                best_score,
            )
            save_confusion_matrix_plot(
                val_metrics["confusion_matrix"],
                output_dir / "val_confusion_matrix.png",
            )
        else:
            bad_epochs += 1

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            class_weights,
            val_metrics,
            best_score,
        )
        history.append({"epoch": epoch, "metrics": val_metrics, "best_score": best_score})
        save_json({"history": history, "class_counts": class_counts.tolist()}, output_dir / "history.json")

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_kappa={val_metrics['cohen_kappa']:.4f} "
            f"best_{monitor}={best_score:.4f}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} epochs without {monitor} improvement.")
            break

    return {"best_score": best_score, "history": history}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EepyNet on Sleep-EDF chunks.")
    parser.add_argument("--config", default="configs/eepynet.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
