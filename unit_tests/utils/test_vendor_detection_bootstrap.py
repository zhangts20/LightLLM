"""Vendor probes must work before platform selection; all runtime APIs are mocked."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def devices():
    name = "device_utils_bootstrap_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "lightllm/utils/device_utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "available,cuda,maca,musa,hip,expected",
    [
        (True, "12.8", None, None, None, True),
        (True, "12.8", "3.8.2.2", None, None, False),
        (True, "12.8", None, "5.2", None, False),
        (True, "12.8", None, None, "6.4", False),
        (False, None, None, None, None, False),
    ],
)
def test_nvidia_probe_does_not_construct_backend(monkeypatch, devices, available, cuda, maca, musa, hip, expected):
    monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: available)
    for name, value in (("cuda", cuda), ("maca", maca), ("musa", musa), ("hip", hip)):
        monkeypatch.setattr(devices.torch.version, name, value, raising=False)

    def unconfigured_backend():
        raise AssertionError("Vendor discovery must not require a selected backend")

    monkeypatch.setattr(devices, "get_backend", unconfigured_backend)
    assert devices.is_nvidia() is expected
    assert devices.is_nvidia() is expected  # cached queries retain the same contract


@pytest.mark.parametrize(
    "selected,vendor_label,expected", [("maca", None, True), ("cuda", "3.8", False), ("mlu", None, False)]
)
def test_maca_dispatch_uses_selected_platform(monkeypatch, devices, selected, vendor_label, expected):
    monkeypatch.setattr(devices.torch.version, "maca", vendor_label, raising=False)
    monkeypatch.setattr(devices, "get_backend", lambda: SimpleNamespace(name=selected))
    assert devices.is_maca() is expected
