# eepynet

EepyNet training pipeline for 5-class sleep staging on PhysioNet Sleep-EDF Expanded EEG.

The repo uses `uv` for environment management. The default sync installs the CPU PyTorch build through the default `torch-cpu` dependency group.

```powershell
uv sync
uv run eepynet-preprocess --config configs/eepynet.yaml
uv run eepynet-make-splits --config configs/eepynet.yaml
uv run eepynet-train --config configs/eepynet.yaml
uv run eepynet-eval --config configs/eepynet.yaml --split val
```

To install the CUDA 12.8 PyTorch build on Linux or Windows, disable the default CPU group and enable the `cu128` extra. Use the same flags with `uv run` so uv does not resync back to the default CPU environment.

```powershell
uv sync --no-group torch-cpu --extra cu128
uv run --no-group torch-cpu --extra cu128 eepynet-train --config configs/eepynet.yaml
```

Training uses `training.device: auto` by default, which selects CUDA whenever the installed PyTorch build reports `torch.cuda.is_available()`, otherwise CPU.

Set `training.device: cuda` to require GPU training and fail fast if CUDA is not available.

Default data assumptions:

- raw EDF files are already under `data/`
- PSG files match `*-PSG.edf`
- hypnogram files match `*-Hypnogram.edf`
- EEG channels are `EEG Fpz-Cz` and `EEG Pz-Oz`
- labels are `W`, `N1`, `N2`, `N3`, `REM`, with stage 4 merged into `N3`

We export native PyTorch checkpoints (`.pt`) for ongoing development. ONNX export is intentionally left for a later portable inference milestone.
