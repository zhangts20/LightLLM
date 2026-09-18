"""MACA regression: mixed full/tail row tiles must overwrite the entire output."""
import pytest
import torch

if not torch.cuda.is_available() or "metax" not in torch.cuda.get_device_name().lower():
    pytest.skip("requires a MetaX GPU", allow_module_level=True)

from lightllm.common.quantization.w8a8 import (
    w8a8MacaQuantizationMethod, w8a8QuantizationMethod, w8a8NPUQuantizationMethod,
    QUANTMETHODS, WeightPack, vllm_ops,
)


def test_platform_registration():
    methods = QUANTMETHODS._quant_methods
    assert methods["w8a8-vllm"]["maca"] is w8a8MacaQuantizationMethod
    assert methods["w8a8-vllm"]["cuda"] is w8a8QuantizationMethod
    assert methods["w8a8"]["ascend"] is w8a8NPUQuantizationMethod


@pytest.mark.parametrize("rows", [30, 143, 255, 256, 257, 293, 362, 511, 512, 513, 1025])
@pytest.mark.parametrize("in_features", [2560, 6144])
def test_partial_rows_against_dequantized_reference(rows, in_features):
    torch.manual_seed(71)
    x = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda")
    weight = torch.randint(-127, 128, (5120, in_features), dtype=torch.int8, device="cuda")
    scales = torch.rand(5120, device="cuda") * .002 + .001
    bias = torch.randn(5120, device="cuda", dtype=x.dtype) * .01
    pack = WeightPack(weight=weight, weight_scale=scales)
    out = torch.full((rows, 5120), 12345., dtype=x.dtype, device=x.device)
    method = w8a8MacaQuantizationMethod.__new__(w8a8MacaQuantizationMethod)
    actual = method.apply(x, pack, out=out, bias=bias)
    q, scale, _ = vllm_ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
    expected = (q.float() @ weight.float().t()) * scale * scales[None, :] + bias.float()
    assert actual is out
    torch.testing.assert_close(out.float(), expected, rtol=.005, atol=.03)


@pytest.mark.parametrize("rows", [30, 293, 362])
def test_graph_replay_overwrites_previous_output(rows):
    torch.manual_seed(19)
    x = torch.randn(rows, 6144, dtype=torch.bfloat16, device="cuda")
    pack = WeightPack(
        weight=torch.randint(-127, 128, (5120, 6144), dtype=torch.int8, device="cuda"),
        weight_scale=torch.full((5120,), .002, device="cuda"),
    )
    out = torch.empty(rows, 5120, dtype=x.dtype, device=x.device)
    method = w8a8MacaQuantizationMethod.__new__(w8a8MacaQuantizationMethod)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        method.apply(x, pack, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        method.apply(x, pack, out=out)
    x.normal_()
    out.fill_(12345.)
    graph.replay()
    q, scale, _ = vllm_ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
    expected = (q.float() @ pack.weight.float().t()) * scale * pack.weight_scale[None, :]
    torch.testing.assert_close(out.float(), expected, rtol=.005, atol=.03)
