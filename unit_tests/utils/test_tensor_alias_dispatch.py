"""Device dispatch contracts exercised with CPU-only tensor and operator mocks."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from lightllm.utils import device_utils, tensor_utils


@pytest.mark.parametrize(
    "device_type,corex,expected",
    [
        ("npu", False, "npu"),
        ("musa", False, "native"),
        ("mlu", False, "native"),
        ("cuda", True, "native"),
        ("cuda", False, "pointer"),
    ],
)
def test_alias_dispatch_preserves_platform_paths(monkeypatch, device_type, corex, expected):
    calls = []
    tensor = SimpleNamespace(
        device=SimpleNamespace(type=device_type),
        dtype="mock_dtype",
        shape=(2, 3),
        is_contiguous=lambda: True,
        data_ptr=lambda: 4096,
        numel=lambda: 6,
        element_size=lambda: 2,
    )
    alias = SimpleNamespace()

    def corex_probe():
        assert device_type == "cuda", "Non-CUDA devices must not enter the CoreX probe"
        calls.append("corex_probe")
        return corex

    def native_alias(actual):
        assert actual is tensor
        calls.append("native")
        return alias

    def npu_alias(actual):
        assert actual is tensor
        calls.append("npu")
        return alias

    class PointerAlias:
        def view(self, **kwargs):
            calls.append(kwargs)
            return self

    pointer_alias = PointerAlias()

    def pointer_view(pointer, nbytes):
        assert (pointer, nbytes) == (4096, 12)
        calls.append("pointer")
        return pointer_alias

    npu = ModuleType("torch_npu")
    npu.__path__ = []
    npu_c = ModuleType("torch_npu._C")
    npu_c._weak_ref_tensor = npu_alias
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setitem(sys.modules, "torch_npu._C", npu_c)
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(tensor_utils, "is_corex", corex_probe)
    monkeypatch.setattr(tensor_utils, "_musa_no_ref_tensor", native_alias)
    monkeypatch.setattr(tensor_utils, "ptr_to_tensor", pointer_view)

    result = tensor_utils.tensor_to_no_ref_tensor(tensor)

    if expected == "pointer":
        assert result is pointer_alias
        assert calls == ["corex_probe", "pointer", {"dtype": "mock_dtype"}, {"size": (2, 3)}]
    else:
        assert result is alias
        assert calls == (["corex_probe"] if device_type == "cuda" else []) + [expected]


def test_maca_pointer_view_keeps_raw_pointer_without_cupy(monkeypatch):
    calls = []
    sentinel = object()

    def as_tensor(holder, **kwargs):
        calls.append((holder.__cuda_array_interface__, kwargs))
        return sentinel

    monkeypatch.setattr(device_utils, "is_maca", lambda: True)
    monkeypatch.setattr(tensor_utils.torch, "as_tensor", as_tensor)
    monkeypatch.setitem(sys.modules, "cupy", None)

    assert tensor_utils.ptr_to_tensor(4096, 12) is sentinel
    interface, kwargs = calls[0]
    assert interface["data"] == (4096, False)
    assert interface["shape"] == (12,)
    assert kwargs == {"device": "cuda"}


def test_noncontiguous_tensor_rejected_before_dispatch():
    tensor = SimpleNamespace(is_contiguous=lambda: False)
    with pytest.raises(AssertionError, match="Only support contiguous tensor"):
        tensor_utils.tensor_to_no_ref_tensor(tensor)
