"""Graph input refresh must preserve captured tensor storage and values."""
import pytest
import torch

from lightllm.platform.runtime.cuda import CudaRuntime
from lightllm.platform.runtime.maca import MacaRuntime


@pytest.mark.parametrize("runtime_class", [CudaRuntime, MacaRuntime])
def test_copy_preserves_graph_storage_and_handles_casts(runtime_class):
    runtime = runtime_class()
    destinations = [torch.empty(2, 3), torch.empty(3, dtype=torch.int32)]
    sources = [torch.arange(6).reshape(2, 3), torch.tensor([2, 7, 11], dtype=torch.int64)]
    pointers = [tensor.data_ptr() for tensor in destinations]
    original_is_cuda = torch.Tensor.is_cuda
    runtime.copy_tensors(destinations, sources)
    assert [tensor.data_ptr() for tensor in destinations] == pointers
    for destination, source in zip(destinations, sources):
        torch.testing.assert_close(destination, source.to(destination.dtype))
    assert torch.Tensor.is_cuda is original_is_cuda


@pytest.mark.parametrize("runtime_class", [CudaRuntime, MacaRuntime])
def test_copy_rejects_count_mismatch(runtime_class):
    with pytest.raises(ValueError, match="counts must match"):
        runtime_class().copy_tensors([torch.empty(1)], [])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA-compatible accelerator")
def test_maca_batched_copy_supports_strides_and_mixed_dtypes():
    runtime = MacaRuntime()
    destinations = [torch.empty(3, 4, device="cuda").t(), torch.empty(2, device="cuda", dtype=torch.int32)]
    sources = [torch.arange(12, device="cuda", dtype=torch.float32).reshape(4, 3), torch.tensor([3, 8], device="cuda")]
    # Two tensors of the same dtype/device also exercise the foreach path.
    destinations.append(torch.empty_like(sources[0]))
    sources.append(sources[0] + 2)
    pointers = [tensor.data_ptr() for tensor in destinations]
    runtime.copy_tensors(destinations, sources)
    assert [tensor.data_ptr() for tensor in destinations] == pointers
    for destination, source in zip(destinations, sources):
        torch.testing.assert_close(destination, source.to(destination.dtype))


def test_native_graph_replay_uses_refreshed_inputs():
    from lightllm.platform import get_backend

    backend = get_backend()
    if backend.name not in {"cuda", "maca", "mlu"} or not backend.runtime.is_available():
        pytest.skip("Requires a supported accelerator runtime")
    runtime = backend.runtime
    device = runtime.target_device(0)
    inputs = [torch.ones(4, device=device), torch.ones(4, device=device)]
    pointers = [tensor.data_ptr() for tensor in inputs]
    for _ in range(2):
        inputs[0] + inputs[1]
    runtime.synchronize()
    graph = backend.graph.create_graph()
    with backend.graph.graph(graph):
        output = inputs[0] + inputs[1]
    for left, right in [(7, 11), (19, 23)]:
        sources = [torch.full_like(inputs[0], left), torch.full_like(inputs[1], right)]
        runtime.copy_tensors(inputs, sources)
        backend.graph.replay_graph(graph)
        runtime.synchronize()
        assert [tensor.data_ptr() for tensor in inputs] == pointers
        torch.testing.assert_close(output, torch.full_like(output, left + right))
