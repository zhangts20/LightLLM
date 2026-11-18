"""Native Ascend ops and storage contracts; run in the TorchNPU environment."""
import gc

import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

from lightllm.platform.runtime.ascend import AscendRuntime
from lightllm.platform.ops import ascend


@pytest.fixture
def device():
    runtime = AscendRuntime()
    runtime.set_device(0)
    return runtime.target_device()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(5, 128), (2, 3, 128)])
def test_layer_norm_matches_cpu_and_preserves_out(device, dtype, shape):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype)
    w = torch.randn(128, dtype=dtype)
    b = torch.randn(128, dtype=dtype)
    expected = torch.nn.functional.layer_norm(x.float(), (128,), w.float(), b.float(), 1e-5).to(dtype)
    out = torch.empty_like(x, device=device)
    pointer = out.data_ptr()
    actual = ascend.layer_norm(input=x.to(device), weight=w.to(device), bias=b.to(device), eps=1e-5, out=out)
    assert actual is out and actual.data_ptr() == pointer
    torch.testing.assert_close(actual.cpu(), expected, atol=0.03, rtol=0.02)


def test_embedding_masks_other_tp_shards(device):
    weight = torch.arange(4 * 128, dtype=torch.float32).reshape(4, 128).to(device)
    ids = torch.tensor([8, 10, 12, 13, 14], device=device)
    actual = ascend.embedding(
        input_ids=ids, weight=weight, vob_start_id=10, vob_end_id=14, out=torch.empty(5, 128, device=device)
    )
    expected = torch.stack([torch.zeros_like(weight[0]), weight[0], weight[2], weight[3], torch.zeros_like(weight[0])])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qk_norm_writes_back_noncontiguous_gqa_views(device, dtype):
    torch.manual_seed(42)
    qkv = torch.randn(3, 2, 8, 128, dtype=dtype, device=device)
    q = qkv[:, :, :4].transpose(0, 1)
    k = qkv[:, :, 4:6].transpose(0, 1)
    v = qkv[:, :, 6:].clone()
    wq = torch.randn(128, dtype=dtype, device=device)
    wk = torch.randn(128, dtype=dtype, device=device)

    def reference(x, w):
        xf = x.cpu().float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * w.cpu().float()).to(dtype)

    expected_q, expected_k = reference(q, wq), reference(k, wk)
    pq, pk = q.data_ptr(), k.data_ptr()
    actual_q, actual_k = ascend.qk_rms_norm(q=q, k=k, w_q=wq, w_k=wk, eps=1e-6)
    assert actual_q.data_ptr() == pq and actual_k.data_ptr() == pk
    torch.testing.assert_close(q.cpu(), expected_q, atol=0.03, rtol=0.02)
    torch.testing.assert_close(k.cpu(), expected_k, atol=0.03, rtol=0.02)
    torch.testing.assert_close(qkv[:, :, 6:], v)


def test_npu_weak_reference_preserves_view_and_device(device):
    from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor

    owner = torch.arange(24, device=device).reshape(4, 6)
    view = tensor_to_no_ref_tensor(owner)
    assert view.device == owner.device
    assert view.data_ptr() == owner.data_ptr()
    assert view.stride() == owner.stride()
    owner.add_(3)
    torch.testing.assert_close(view, owner)


def test_storage_count_compatibility_tracks_live_aliases(device, monkeypatch):
    import torch_npu
    import lightllm.platform
    from types import SimpleNamespace
    from lightllm.common.basemodel.layer_infer.cache_tensor_manager import CacheTensorManager

    monkeypatch.delattr(torch_npu._C, "_storage_Use_Count", raising=False)
    monkeypatch.setattr(lightllm.platform, "get_backend", lambda: SimpleNamespace(name="ascend"))
    manager = CacheTensorManager()
    manager._ensure_use_count()
    x = torch.ones(8, device=device)
    storage = x.untyped_storage()._weak_ref()
    try:
        before = manager.use_count(storage)
        alias = x.view(2, 4)
        assert manager.use_count(storage) > before
        del alias
        gc.collect()
        assert manager.use_count(storage) == before
    finally:
        torch.UntypedStorage._free_weak_ref(storage)


def test_native_attention_prefill_and_decode_match_cpu(device):
    from lightllm.utils.backend_validator import _validate_ascend_fia

    success, error = _validate_ascend_fia()
    assert success, error
