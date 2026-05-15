# eepynet

EepyNet training pipeline for 5-class sleep staging on PhysioNet Sleep-EDF Expanded EEG.

The repo uses `uv` for environment management. Choose one PyTorch backend during sync, then use normal `uv run` commands.

```powershell
uv sync --extra cpu
uv run eepynet-preprocess --config configs/eepynet.yaml
uv run eepynet-make-splits --config configs/eepynet.yaml
uv run eepynet-train --config configs/eepynet.yaml
uv run eepynet-eval --config configs/eepynet.yaml --split val
```

To install the CUDA 13.2 PyTorch build on Linux or Windows for recent NVIDIA GPUs, use the `cu132` extra.

```powershell
uv sync --extra cu132
uv run eepynet-train --config configs/eepynet.yaml
```

CUDA 12.6 remains available as `cu126` for users who need the legacy wheel target for older supported GPU architectures:

```powershell
uv sync --extra cu126
uv run eepynet-train --config configs/eepynet.yaml
```

The `cpu`, `cu126`, and `cu132` extras are mutually exclusive. Running plain `uv sync` later will remove the selected PyTorch backend, so use one of the backend extras whenever you intentionally resync the environment.

Training uses `training.device: auto` by default, which selects CUDA whenever the installed PyTorch build reports `torch.cuda.is_available()`, otherwise CPU.

Set `training.device: cuda` to require GPU training and fail fast if CUDA is not available.

Default data assumptions:

- raw EDF files are already under `data/`
- PSG files match `*-PSG.edf`
- hypnogram files match `*-Hypnogram.edf`
- EEG channels are `EEG Fpz-Cz` and `EEG Pz-Oz`
- labels are `W`, `N1`, `N2`, `N3`, `REM`, with stage 4 merged into `N3`

We export native PyTorch checkpoints (`.pt`) for ongoing development. ONNX export is intentionally left for a later portable inference milestone.
