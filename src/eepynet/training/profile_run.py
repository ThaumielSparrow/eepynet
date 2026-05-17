"""Profile a training step and print a breakdown of where time is spent.

Usage:
    uv run eepynet-profile                            # 3 warmup + 5 profiled steps
    uv run eepynet-profile --warmup-steps 5           # longer warmup (needed with compile)
    uv run eepynet-profile --trace trace.json         # also export a chrome://tracing file
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import WeightedRandomSampler

from eepynet.config import load_config
from eepynet.data.augment import SignalAugmenter
from eepynet.data.dataset import compute_chunk_sampling_weights, compute_class_counts
from eepynet.models.eepynet import EepyNet
from eepynet.training.losses import class_weights_from_counts, masked_focal_loss, masked_weighted_cross_entropy
from eepynet.training.trainer import build_dataloader, build_dataset, move_batch_to_device
from eepynet.utils import get_torch_device, load_json, seed_everything


def _prefetch(loader: Any, n: int) -> list:
    """Pull n batches off the loader (cycling if the dataset is smaller) before profiling starts."""
    batches = []
    it = iter(loader)
    for _ in range(n):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader)
            batches.append(next(it))
    return batches


def _step(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,  # pyright: ignore[reportPrivateImportUsage]
    device: torch.device,
    class_weights: torch.Tensor,
    use_amp: bool,
    amp_dtype: torch.dtype,
    label_smoothing: float,
    focal_gamma: float = 0.0,
) -> float:
    x, y, mask = move_batch_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with record_function("forward"):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            if focal_gamma > 0:
                loss = masked_focal_loss(logits, y, mask, class_weights, gamma=focal_gamma)
            else:
                loss = masked_weighted_cross_entropy(
                    logits, y, mask, class_weights, label_smoothing=label_smoothing
                )
    with record_function("backward"):
        scaler.scale(loss).backward()
    with record_function("optimizer_step"):
        scaler.step(optimizer)
        scaler.update()
    return float(loss.detach())


def run_profile(
    config: dict[str, Any],
    warmup_steps: int,
    profile_steps: int,
    trace_path: str | None,
    row_limit: int,
) -> None:
    seed_everything(int(config["training"]["seed"]))

    device = get_torch_device(config["training"].get("device", "auto"))
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype_name = str(config["training"].get("amp_dtype", "bf16")).lower()
    amp_dtype = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16,
    }[amp_dtype_name]
    compile_mode = str(config["training"].get("compile_mode", "off")).lower()
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))
    focal_gamma = float(config["training"].get("focal_gamma", 0.0))
    class_weight_power = float(config["training"].get("class_weight_power", 1.0))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU : {torch.cuda.get_device_name(device)}")

    print(f"device={device} | amp={use_amp} dtype={amp_dtype_name} | compile_mode={compile_mode!r}")
    print(f"warmup={warmup_steps} steps | profiling={profile_steps} steps")
    if compile_mode != "off" and warmup_steps < 2:
        print("  WARNING: compile_mode is on — first step will be slow (JIT compile). Consider --warmup-steps 3+.")
    print()

    # ---- data ---------------------------------------------------------------
    split_manifest = load_json(config["paths"]["split_path"])

    class_counts = compute_class_counts(
        config["paths"]["processed_dir"],
        split_manifest,
        split="train",
        num_classes=int(config["model"]["num_classes"]),
    )
    class_weights_cpu = class_weights_from_counts(class_counts, power=class_weight_power)
    class_weights = class_weights_cpu.to(device)

    augmenter = SignalAugmenter.from_config(config)
    if augmenter is not None:
        print(f"Augmentation: {augmenter}")
    train_dataset = build_dataset(config, "train", augmenter=augmenter)

    train_sampler = None
    if bool(config["training"].get("use_weighted_sampler", False)):
        chunk_weights = compute_chunk_sampling_weights(train_dataset, class_weights_cpu)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(chunk_weights, dtype=torch.double),
            num_samples=len(chunk_weights),
            replacement=True,
        )

    loader = build_dataloader(config, train_dataset, shuffle=True, sampler=train_sampler)

    # Pre-fetch all batches we need so disk I/O (and augmentation CPU work)
    # don't distort the GPU step profile.
    total = warmup_steps + profile_steps
    print(f"Pre-fetching {total} batches from train loader...")
    batches = _prefetch(loader, total)
    print("Done.\n")

    # ---- model --------------------------------------------------------------
    base_model = EepyNet(**config["model"]).to(device)
    if bool(config["training"].get("use_gradient_checkpointing", False)):
        base_model.epoch_encoder.use_checkpoint = True

    model: torch.nn.Module
    if compile_mode == "off" or device.type != "cuda":
        model = base_model
    else:
        print(f"Compiling model (mode={compile_mode!r}) — first warmup step will be slow...")
        model = torch.compile(base_model, mode=compile_mode, dynamic=True)

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)  # pyright: ignore[reportPrivateImportUsage]

    model.train()

    # ---- warmup -------------------------------------------------------------
    step_kwargs = dict(
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        class_weights=class_weights,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
    )

    print("Warmup steps (not profiled):")
    for i, batch in enumerate(batches[:warmup_steps]):
        t0 = time.perf_counter()
        loss = _step(model, batch, **step_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        note = " ← torch.compile JIT" if compile_mode != "off" and i == 0 else ""
        print(f"  step {i + 1:2d}: {elapsed_ms:7.0f} ms  loss={loss:.4f}{note}")
    print()

    # ---- profiled steps -----------------------------------------------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    activities = (
        [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if device.type == "cuda"
        else [ProfilerActivity.CPU]
    )

    print(f"Profiling {profile_steps} steps...")
    step_times: list[float] = []

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for batch in batches[warmup_steps:]:
            t0 = time.perf_counter()
            _step(model, batch, **step_kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_times.append((time.perf_counter() - t0) * 1000)
            prof.step()

    # ---- summary header -----------------------------------------------------
    avg_ms = sum(step_times) / len(step_times)
    print(f"  per-step times : {' '.join(f'{t:.0f}ms' for t in step_times)}")
    print(f"  mean step time : {avg_ms:.0f} ms  ({1000 / avg_ms:.2f} steps/s)")

    if device.type == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 ** 2
        print(f"  peak VRAM     : {peak_alloc_mb:.0f} MB allocated / {peak_reserved_mb:.0f} MB reserved")
    print()

    # ---- profiler tables ----------------------------------------------------
    # torch 2.12 uses self_device_time_total on FunctionEventAvg; the string
    # "self_cuda_time_total" still works as a table() sort key but not as an attribute.
    _dev_time = lambda e: getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
    sort_by = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    averages = prof.key_averages()

    sep = "=" * 160

    print(sep)
    print("TOP OPS — by self CUDA time  (where the GPU is actually spending its time)")
    print(sep)
    print(averages.table(sort_by=sort_by, row_limit=row_limit))

    print(sep)
    print("STEP PHASES — forward / backward / optimizer_step")
    print(sep)
    phase_rows = [e for e in averages if e.key in {"forward", "backward", "optimizer_step"}]
    if phase_rows:
        cuda_total_us = sum(_dev_time(e) for e in averages)
        header = f"{'Phase':<20} {'CUDA time':>12} {'% total':>8} {'CPU time':>12} {'calls':>6}"
        print(header)
        print("-" * len(header))
        for row in sorted(phase_rows, key=_dev_time, reverse=True):
            pct = 100.0 * _dev_time(row) / cuda_total_us if cuda_total_us > 0 else 0
            print(
                f"{row.key:<20} "
                f"{_dev_time(row) / 1000:>10.1f}ms "
                f"{pct:>7.1f}% "
                f"{row.self_cpu_time_total / 1000:>10.1f}ms "
                f"{row.count:>6}"
            )
    else:
        print("(no phase annotations found — record_function markers may not have fired)")
    print()

    if trace_path:
        out = Path(trace_path)
        prof.export_chrome_trace(str(out))
        print(f"Chrome trace → {out.resolve()}")
        print("  Open at chrome://tracing  or  https://ui.perfetto.dev")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profile an EepyNet training step and show where time is spent."
    )
    p.add_argument("--config", default="configs/eepynet.yaml")
    p.add_argument(
        "--warmup-steps", type=int, default=3,
        help="Steps run before profiling — lets torch.compile finish. Default: 3.",
    )
    p.add_argument(
        "--profile-steps", type=int, default=5,
        help="Steps to profile. Default: 5.",
    )
    p.add_argument(
        "--trace", default=None, metavar="PATH",
        help="Export a chrome trace JSON for deeper inspection (chrome://tracing).",
    )
    p.add_argument(
        "--row-limit", type=int, default=25,
        help="Rows in the top-ops table. Default: 25.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    run_profile(config, args.warmup_steps, args.profile_steps, args.trace, args.row_limit)


if __name__ == "__main__":
    main()
