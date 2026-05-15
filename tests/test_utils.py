import pytest
import torch

from eepynet.utils import get_torch_device


def test_get_torch_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert get_torch_device("auto").type == "cuda"


def test_get_torch_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_torch_device("auto").type == "cpu"


def test_get_torch_device_cuda_requires_available_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="uv sync --extra cu132"):
        get_torch_device("cuda")
