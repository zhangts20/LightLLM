"""Device numerical regressions for upstream precision fix 8e1ddf29.

Run with: pytest -q unit_tests/common/basemodel/triton_kernel/linear_att/test_mtp_precision.py
CUDA also covers the MACA CUDA-compatible runtime. NPU cases require torch_npu.
"""

import pytest
import torch


@pytest.fixture(params=["cuda", "npu"])
def device(request):
    if request.param == "npu":
        pytest.importorskip("torch_npu")
    runtime = getattr(torch, request.param, None)
    if runtime is None or not runtime.is_available():
        pytest.skip(f"{request.param} accelerator required")
    return request.param


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/MACA required")
def test_bf16_conv_product_cancellation():
    from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import causal_conv1d_update

    # BF16-exact operands: (1 + 2**-7)**2 - (1 + 2**-6) = 2**-14.
    # A BF16 intermediate product rounds away that residual entirely.
    x = torch.ones((2, 64), dtype=torch.bfloat16, device="cuda")
    weight = torch.tensor([1.0078125, -1.015625], dtype=x.dtype, device=x.device).repeat(64, 1)
    state = torch.full((1, 64, 2), 1.0078125, dtype=x.dtype, device=x.device)
    output = causal_conv1d_update(
        x=x, conv_state=state, weight=weight, mtp_step=1,
        conv_state_indices=torch.tensor([0], dtype=torch.int32, device=x.device),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=x.device),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=x.device),
        activation=None,
    )
    expected = torch.tensor([2**-14, -2**-7], dtype=x.dtype, device=x.device)[:, None].expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [2, 4])
def test_mtp_recurrence_matches_serial_persistent_state(device, cache_dtype, tokens):
    from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
        mtp_fused_recurrent_gated_delta_rule,
    )

    # Generate identical CPU fixtures for every platform. Use non-overlapping
    # input/output slots so this test isolates precision rather than aliasing.
    generator = torch.Generator().manual_seed(1569)

    def randn(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)

    q, k = randn((1, tokens, 2, 64)), randn((1, tokens, 2, 64))
    v = randn((1, tokens, 4, 64))
    a, b = randn((tokens, 4)), randn((tokens, 4))
    A_log, dt_bias = randn((4,), torch.float32) * 0.1, randn((4,), torch.float32) * 0.1
    cache = randn((tokens + 1, 4, 64, 64), cache_dtype)
    speculative, serial = cache.clone(), cache.clone()

    def run(start, end, states, read_slot, write_slots):
        read_indices = torch.full((1, 1), read_slot, dtype=torch.int32, device=device)
        if device == "npu":
            read_indices = read_indices[:, 0].contiguous()
        return mtp_fused_recurrent_gated_delta_rule(
            q=q[:, start:end], k=k[:, start:end], v=v[:, start:end],
            initial_state=states, final_state=states,
            cu_seqlens=torch.tensor([0, end - start], dtype=torch.int32, device=device),
            ssm_state_indices=read_indices,
            ssm_state_write_indices=torch.tensor([write_slots], dtype=torch.int32, device=device),
            num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
            A_log=A_log, dt_bias=dt_bias, a_raw=a[start:end], b_raw=b[start:end],
        )[0]

    actual = run(0, tokens, speculative, tokens, list(range(tokens)))
    expected = torch.cat(
        [run(i, i + 1, serial, tokens if i == 0 else i - 1, [i]) for i in range(tokens)], dim=1
    )
    # Each one-token launch reloads the persisted cache dtype. Multi-token
    # verification must produce the same outputs AND every saved token state.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(speculative[:tokens], serial[:tokens], rtol=0, atol=0)
    torch.testing.assert_close(speculative[tokens], cache[tokens], rtol=0, atol=0)
