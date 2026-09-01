"""Ascend Triton MTP causal convolution with fused compact-state update.

Adapter from the SGL kernel implementation at:
https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/sgl_kernel_npu/mamba/causal_conv1d.py
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_mtp_compact_sgl_npu_kernel(
    x,
    weight,
    bias,
    state,
    accepted,
    output,
    dim: tl.constexpr,
    stride_x_batch: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_state_batch: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_o_batch: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_o_dim: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    TOKENS_PER_SEQ: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    channel_mask = channel < dim

    weight_base = weight + channel * stride_w_dim
    weight0 = tl.load(weight_base, mask=channel_mask, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 2:
        weight1 = tl.load(weight_base + stride_w_width, mask=channel_mask, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        weight2 = tl.load(weight_base + 2 * stride_w_width, mask=channel_mask, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        weight3 = tl.load(weight_base + 3 * stride_w_width, mask=channel_mask, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        weight4 = tl.load(weight_base + 4 * stride_w_width, mask=channel_mask, other=0.0).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        weight5 = tl.load(weight_base + 5 * stride_w_width, mask=channel_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        bias_value = tl.load(bias + channel, mask=channel_mask, other=0.0).to(tl.float32)
    else:
        bias_value = tl.zeros((BLOCK_N,), dtype=tl.float32)

    accepted_offset = (tl.load(accepted + batch_idx) - 1).to(tl.int64)
    state_base = state + batch_idx * stride_state_batch + channel * stride_state_dim
    initial_state = state_base + accepted_offset * stride_state_token

    raw0 = tl.load(initial_state, mask=channel_mask, other=0.0)
    history0 = raw0.to(tl.float16)
    if KERNEL_WIDTH >= 3:
        raw1 = tl.load(initial_state + stride_state_token, mask=channel_mask, other=0.0)
        history1 = raw1.to(tl.float16)
        tl.store(state_base, raw1, mask=channel_mask)
    if KERNEL_WIDTH >= 4:
        raw2 = tl.load(initial_state + 2 * stride_state_token, mask=channel_mask, other=0.0)
        history2 = raw2.to(tl.float16)
        tl.store(state_base + stride_state_token, raw2, mask=channel_mask)
    if KERNEL_WIDTH >= 5:
        raw3 = tl.load(initial_state + 3 * stride_state_token, mask=channel_mask, other=0.0)
        history3 = raw3.to(tl.float16)
        tl.store(state_base + 2 * stride_state_token, raw3, mask=channel_mask)
    if KERNEL_WIDTH >= 6:
        raw4 = tl.load(initial_state + 4 * stride_state_token, mask=channel_mask, other=0.0)
        history4 = raw4.to(tl.float16)
        tl.store(state_base + 3 * stride_state_token, raw4, mask=channel_mask)

    x_base = x + batch_idx * stride_x_batch + channel * stride_x_dim
    output_base = output + batch_idx * stride_o_batch + channel * stride_o_dim

    for token_idx in tl.static_range(TOKENS_PER_SEQ):
        current_raw = tl.load(
            x_base + token_idx * stride_x_token,
            mask=channel_mask,
            other=0.0,
        )
        current = current_raw.to(tl.float16)
        acc = bias_value
        if KERNEL_WIDTH == 2:
            acc += history0.to(tl.float32) * weight0 + current.to(tl.float32) * weight1
            history0 = current
        elif KERNEL_WIDTH == 3:
            acc += (
                history0.to(tl.float32) * weight0
                + history1.to(tl.float32) * weight1
                + current.to(tl.float32) * weight2
            )
            history0 = history1
            history1 = current
        elif KERNEL_WIDTH == 4:
            acc += (
                history0.to(tl.float32) * weight0
                + history1.to(tl.float32) * weight1
                + history2.to(tl.float32) * weight2
                + current.to(tl.float32) * weight3
            )
            history0 = history1
            history1 = history2
            history2 = current
        elif KERNEL_WIDTH == 5:
            acc += (
                history0.to(tl.float32) * weight0
                + history1.to(tl.float32) * weight1
                + history2.to(tl.float32) * weight2
                + history3.to(tl.float32) * weight3
                + current.to(tl.float32) * weight4
            )
            history0 = history1
            history1 = history2
            history2 = history3
            history3 = current
        elif KERNEL_WIDTH == 6:
            acc += (
                history0.to(tl.float32) * weight0
                + history1.to(tl.float32) * weight1
                + history2.to(tl.float32) * weight2
                + history3.to(tl.float32) * weight3
                + history4.to(tl.float32) * weight4
                + current.to(tl.float32) * weight5
            )
            history0 = history1
            history1 = history2
            history2 = history3
            history3 = history4
            history4 = current

        # Match the upstream NPU kernel's FP16 intermediate rounding before
        # applying SiLU, while retaining a BF16 output/state cache.
        acc = acc.to(tl.float16, fp_downcast_rounding="rtne")
        if SILU_ACTIVATION:
            acc_fp32 = acc.to(tl.float32)
            acc = acc_fp32 / (1.0 + tl.exp(-acc_fp32))
        tl.store(output_base + token_idx * stride_o_token, acc, mask=channel_mask)

        state_token = KERNEL_WIDTH - 2 + token_idx
        tl.store(state_base + state_token * stride_state_token, current_raw, mask=channel_mask)


def causal_conv1d_update_sgl_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight_t: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    conv_state_indices: torch.Tensor,
    mtp_step: int,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    assert weight_t.ndim == 2 and weight_t.is_contiguous()

    original_dtype = x.dtype
    x = x.to(conv_state.dtype)
    batch = conv_state_indices.numel()
    tokens_per_seq = mtp_step + 1
    dim = x.shape[1]
    width = weight_t.shape[0]
    state_len = conv_state.shape[2]
    assert 2 <= width <= 6

    # The upstream kernel is fast only when channels are contiguous in both
    # state and weight.  Compact active slots and transpose once around the
    # fused kernel; the persistent request cache keeps LightLLM's layout.
    compact_state = (
        torch.index_select(conv_state, 0, conv_state_indices)
        .transpose(1, 2)
        .contiguous()
    )
    x_3d = x.view(batch, tokens_per_seq, dim)
    output = x_3d

    grid = (batch, triton.cdiv(dim, 512))
    _causal_conv1d_mtp_compact_sgl_npu_kernel[grid](
        x_3d,
        weight_t,
        bias,
        compact_state,
        num_accepted_tokens,
        output,
        dim=dim,
        stride_x_batch=x_3d.stride(0),
        stride_x_token=x_3d.stride(1),
        stride_x_dim=x_3d.stride(2),
        stride_w_width=weight_t.stride(0),
        stride_w_dim=weight_t.stride(1),
        stride_state_batch=compact_state.stride(0),
        stride_state_token=compact_state.stride(1),
        stride_state_dim=compact_state.stride(2),
        stride_o_batch=output.stride(0),
        stride_o_token=output.stride(1),
        stride_o_dim=output.stride(2),
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        TOKENS_PER_SEQ=tokens_per_seq,
        SILU_ACTIVATION=activation in ("silu", "swish"),
        BLOCK_N=512,
        num_stages=1,
        multibuffer=False,
    )

    conv_state.index_copy_(0, conv_state_indices, compact_state.transpose(1, 2))
    return output.view_as(x).to(original_dtype)
