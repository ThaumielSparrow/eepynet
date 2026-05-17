from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import medfilt
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from eepynet.config import load_config
from eepynet.constants import CLASS_NAMES
from eepynet.data.augment import SignalAugmenter
from eepynet.data.dataset import (
    SleepEDFChunkDataset,
    compute_chunk_sampling_weights,
    compute_class_counts,
)
from eepynet.models.eepynet import EepyNet
from eepynet.training.losses import class_weights_from_counts, masked_focal_loss, masked_weighted_cross_entropy
from eepynet.training.metrics import compute_metrics, save_confusion_matrix_plot
from eepynet.utils import ensure_dir, get_torch_device, load_json, save_json, seed_everything


def build_dataset(
    config: dict[str, Any],
    split: str,
    augmenter: SignalAugmenter | None = None,
) -> SleepEDFChunkDataset:
    dataset_cfg = config["dataset"]
    stride_key = "train_stride" if split == "train" else "eval_stride"
    return SleepEDFChunkDataset(
        processed_dir=config["paths"]["processed_dir"],
        split_manifest=config["paths"]["split_path"],
        split=split,
        epochs_per_chunk=int(dataset_cfg["epochs_per_chunk"]),
        stride=int(dataset_cfg[stride_key]),
        augmenter=augmenter,
    )


def build_dataloader(
    config: dict[str, Any],
    dataset: SleepEDFChunkDataset,
    shuffle: bool,
    sampler: Sampler | None = None,
    drop_last: bool = False,
) -> DataLoader:
    num_workers = int(config["dataset"].get("num_workers", 0))
    prefetch_factor = int(config["dataset"].get("prefetch_factor", 2)) if num_workers > 0 else None
    # pin_memory only helps when workers are running (enables non-blocking H2D transfers
    # while the next batch is being prefetched). With num_workers=0 it just makes a second
    # locked copy of the batch for no gain, burning an extra 117 MB of unswappable RAM.
    pin_memory = torch.cuda.is_available() and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
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
    amp_dtype: torch.dtype = torch.float16,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x, y, mask = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            if focal_gamma > 0:
                loss = masked_focal_loss(logits, y, mask, class_weights, gamma=focal_gamma)
            else:
                loss = masked_weighted_cross_entropy(
                    logits, y, mask, class_weights, label_smoothing=label_smoothing
                )
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
    amp_dtype: torch.dtype = torch.float16,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    smooth_window: int = 1,
) -> dict[str, Any]:
    """Evaluate with per-(record, epoch) softmax averaging across overlapping chunks.

    ``smooth_window``: if > 1 and odd, applies a per-record median filter over the
    argmax predictions before scoring — collapses isolated stage mispredictions.
    """

    model.eval()
    dataset = loader.dataset
    if not isinstance(dataset, SleepEDFChunkDataset):
        raise TypeError("evaluate_model requires a SleepEDFChunkDataset")
    num_classes = int(model.model_config["num_classes"])

    prob_sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    record_labels: dict[str, np.ndarray] = {}
    losses: list[float] = []

    for batch in loader:
        x, y, mask = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            if focal_gamma > 0:
                loss = masked_focal_loss(logits, y, mask, class_weights, gamma=focal_gamma)
            else:
                loss = masked_weighted_cross_entropy(
                    logits, y, mask, class_weights, label_smoothing=label_smoothing
                )
        losses.append(float(loss.detach().cpu()))

        probs_np = logits.softmax(dim=-1).float().detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        starts = batch["start_epoch"]
        nums = batch["num_epochs"]
        starts = starts.cpu().numpy() if torch.is_tensor(starts) else np.asarray(starts)
        nums = nums.cpu().numpy() if torch.is_tensor(nums) else np.asarray(nums)
        rids = batch["record_id"]

        for i, rid in enumerate(rids):
            s = int(starts[i])
            n = int(nums[i])
            if rid not in prob_sums:
                y_arr = np.asarray(
                    np.load(dataset.processed_dir / rid / "y.npy", mmap_mode="r"),
                    dtype=np.int64,
                ).copy()
                total_epochs = int(y_arr.shape[0])
                prob_sums[rid] = np.zeros((total_epochs, num_classes), dtype=np.float32)
                counts[rid] = np.zeros((total_epochs,), dtype=np.int32)
                record_labels[rid] = y_arr

            chunk_mask = mask_np[i, :n]
            if not chunk_mask.any():
                continue
            prob_sums[rid][s : s + n] += probs_np[i, :n, :] * chunk_mask[:, None]
            counts[rid][s : s + n] += chunk_mask.astype(np.int32)

    do_smooth = smooth_window > 1 and smooth_window % 2 == 1
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    for rid, ps in prob_sums.items():
        c = counts[rid]
        covered = c > 0
        if not covered.any():
            continue
        mean_probs = ps[covered] / c[covered, None]
        preds = mean_probs.argmax(axis=-1).astype(np.int64)
        if do_smooth:
            preds = medfilt(preds, kernel_size=smooth_window).astype(np.int64)
        labels = record_labels[rid][covered]
        keep = labels >= 0
        if not keep.any():
            continue
        y_true_chunks.append(labels[keep])
        y_pred_chunks.append(preds[keep])

    true = np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
    pred = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)
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
    amp_dtype_name = str(config["training"].get("amp_dtype", "bf16")).lower()
    amp_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}[amp_dtype_name]
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)} | amp={use_amp} dtype={amp_dtype_name}")

    augmenter = SignalAugmenter.from_config(config)
    if augmenter is not None:
        print(f"Augmentation: {augmenter}")
    train_dataset = build_dataset(config, "train", augmenter=augmenter)
    val_dataset = build_dataset(config, "val")

    class_counts = compute_class_counts(
        config["paths"]["processed_dir"],
        split_manifest,
        split="train",
        num_classes=int(config["model"]["num_classes"]),
    )
    class_weight_power = float(config["training"].get("class_weight_power", 1.0))
    class_weights_cpu = class_weights_from_counts(class_counts, power=class_weight_power)
    class_weights = class_weights_cpu.to(device)

    train_sampler: Sampler | None = None
    if bool(config["training"].get("use_weighted_sampler", False)):
        chunk_weights = compute_chunk_sampling_weights(train_dataset, class_weights_cpu)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(chunk_weights, dtype=torch.double),
            num_samples=len(chunk_weights),
            replacement=True,
        )
        print(
            f"WeightedRandomSampler over {len(chunk_weights)} chunks "
            f"(weight min/mean/max = {chunk_weights.min():.3f}/{chunk_weights.mean():.3f}/{chunk_weights.max():.3f})"
        )

    train_loader = build_dataloader(config, train_dataset, shuffle=True, sampler=train_sampler, drop_last=True)
    val_loader = build_dataloader(config, val_dataset, shuffle=False)

    label_smoothing = float(config["training"].get("label_smoothing", 0.0))
    focal_gamma = float(config["training"].get("focal_gamma", 0.0))
    smooth_window = int(config["training"].get("eval_smooth_window", 1))
    loss_desc = f"focal(gamma={focal_gamma})" if focal_gamma > 0 else f"cross_entropy(label_smoothing={label_smoothing})"
    print(
        f"Class weights (power={class_weight_power}): "
        f"{[round(w, 3) for w in class_weights_cpu.tolist()]} | "
        f"loss={loss_desc} | eval_smooth_window={smooth_window}"
    )

    base_model = EepyNet(**config["model"]).to(device)
    if bool(config["training"].get("use_gradient_checkpointing", False)):
        base_model.epoch_encoder.use_checkpoint = True
        print("Gradient checkpointing enabled on EpochSignalEncoder")

    # torch.compile is opt-in via training.compile_mode. Must wrap AFTER the
    # use_checkpoint flag is set so the compiled graph captures the checkpoint
    # hooks. We keep `base_model` around for state_dict / parameters access —
    # the OptimizedModule returned by compile delegates attribute lookups to
    # `_orig_mod`, but using the raw reference avoids version-specific quirks.
    compile_mode = str(config["training"].get("compile_mode", "off")).lower()
    if compile_mode == "off" or device.type != "cuda":
        if compile_mode != "off":
            print(f"compile_mode={compile_mode!r} requested but device={device.type}; running eager.")
        model: torch.nn.Module = base_model
    else:
        print(
            f"Compiling model with torch.compile(mode={compile_mode!r}, dynamic=True). "
            "First train/eval iteration will pause to compile."
        )
        model = torch.compile(base_model, mode=compile_mode, dynamic=True)

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16) # pyright: ignore[reportPrivateImportUsage]

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
            amp_dtype=amp_dtype,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            class_weights=class_weights,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            smooth_window=smooth_window,
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
                base_model,
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
            base_model,
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
