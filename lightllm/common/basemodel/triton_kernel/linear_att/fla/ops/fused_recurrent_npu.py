from __future__ import annotations

import torch
import triton
import triton.language as tl

from .op import exp


@triton.jit
def _prepare_recurrent_qk_npu(
    q,
    k,
    q_out,
    k_out,
    rows,
    H: tl.constexpr,
    K: tl.constexpr,
    STRIDE_Q: tl.constexpr,
    STRIDE_K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(rows, BT), tl.num_programs(0)):
        row = block * BT + tl.arange(0, BT)
        col = tl.arange(0, BK)
        mask = (row[:, None] < rows) & (col[None, :] < K)
        q_val = tl.load(
            q + (row // H)[:, None] * STRIDE_Q + (row % H)[:, None] * K + col[None, :],
            mask=mask,
            other=0,
        ).to(tl.float32)
        k_val = tl.load(
            k + (row // H)[:, None] * STRIDE_K + (row % H)[:, None] * K + col[None, :],
            mask=mask,
            other=0,
        ).to(tl.float32)
        q_val = q_val / tl.sqrt(tl.sum(q_val * q_val, axis=1) + 1e-6)[:, None]
        k_val = k_val / tl.sqrt(tl.sum(k_val * k_val, axis=1) + 1e-6)[:, None]
        tl.store(q_out + row[:, None] * K + col[None, :], q_val, mask=mask)
        tl.store(k_out + row[:, None] * K + col[None, :], k_val, mask=mask)


@triton.jit
def _fused_recurrent_gated_delta_rule_npu_kernel(
    q,
    k,
    v,
    g,
    beta,
    output,
    initial_state,
    state_indices,
    cu_seqlens,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_initial_state: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    value_tile = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence_index = sequence_head // HV
    value_head = sequence_head % HV
    key_head = value_head // (HV // H)

    begin = tl.load(cu_seqlens + sequence_index).to(tl.int64)
    end = tl.load(cu_seqlens + sequence_index + 1).to(tl.int64)
    sequence_length = end - begin

    key_offsets = tl.arange(0, BK)
    value_offsets = value_tile * BV + tl.arange(0, BV)
    key_mask = key_offsets < K
    value_mask = value_offsets < V
    state_mask = key_mask[:, None] & value_mask[None, :]

    q_ptr = q + begin * stride_q_token + key_head * K + key_offsets
    k_ptr = k + begin * stride_k_token + key_head * K + key_offsets
    v_ptr = v + begin * stride_v_token + value_head * V + value_offsets
    g_ptr = g + begin * stride_g_token + value_head
    beta_ptr = beta + begin * stride_beta_token + value_head
    output_ptr = output + begin * stride_output_token + value_head * V + value_offsets

    state_index = tl.load(state_indices + sequence_index)

    state = tl.zeros([BK, BV], dtype=tl.float32)
    initial_ptr = initial_state + state_index * stride_initial_state
    initial_ptr += value_head * K * V + key_offsets[:, None] * V + value_offsets[None, :]
    state += tl.load(initial_ptr, mask=state_mask, other=0).to(tl.float32)

    for _ in range(0, sequence_length):
        query = tl.load(q_ptr, mask=key_mask, other=0).to(tl.float32)
        key = tl.load(k_ptr, mask=key_mask, other=0).to(tl.float32)
        value = tl.load(v_ptr, mask=value_mask, other=0).to(tl.float32)

        if USE_QK_L2NORM:
            query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
            key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        query *= scale

        decay = tl.load(g_ptr).to(tl.float32)
        state *= exp(decay)
        update_scale = tl.load(beta_ptr).to(tl.float32)
        value -= tl.sum(state * key[:, None], axis=0)
        value *= update_scale
        state += key[:, None] * value[None, :]
        result = tl.sum(state * query[:, None], axis=0)
        tl.store(output_ptr, result.to(output_ptr.dtype.element_ty), mask=value_mask)

        q_ptr += stride_q_token
        k_ptr += stride_k_token
        v_ptr += stride_v_token
        g_ptr += stride_g_token
        beta_ptr += stride_beta_token
        output_ptr += stride_output_token

    tl.store(initial_ptr, state.to(initial_ptr.dtype.element_ty), mask=state_mask)


@triton.jit
def _fused_recurrent_gated_delta_rule_npu_safe_kernel(
    q,
    k,
    v,
    g,
    beta,
    output,
    initial_state,
    state_indices,
    cu_seqlens,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_initial_state: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    SEQUENCE_HEADS: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    # Only physical programs are launched. Each logical state tile is updated
    # once, including a partial final wave in the compiler's auto-block mapping.
    for block in range(tl.program_id(0), BLOCKS, tl.num_programs(0)):
        value_tile = block // SEQUENCE_HEADS
        sequence_head = block % SEQUENCE_HEADS
        sequence_index = sequence_head // HV
        value_head = sequence_head % HV
        key_head = value_head // (HV // H)

        begin = tl.load(cu_seqlens + sequence_index).to(tl.int64)
        end = tl.load(cu_seqlens + sequence_index + 1).to(tl.int64)
        sequence_length = end - begin

        key_offsets = tl.arange(0, BK)
        value_offsets = value_tile * BV + tl.arange(0, BV)
        key_mask = key_offsets < K
        value_mask = value_offsets < V
        state_mask = key_mask[:, None] & value_mask[None, :]

        q_ptr = q + begin * stride_q_token + key_head * K + key_offsets
        k_ptr = k + begin * stride_k_token + key_head * K + key_offsets
        v_ptr = v + begin * stride_v_token + value_head * V + value_offsets
        g_ptr = g + begin * stride_g_token + value_head
        beta_ptr = beta + begin * stride_beta_token + value_head
        output_ptr = output + begin * stride_output_token + value_head * V + value_offsets

        state_index = tl.load(state_indices + sequence_index)

        state = tl.zeros([BK, BV], dtype=tl.float32)
        initial_ptr = initial_state + state_index * stride_initial_state
        initial_ptr += value_head * K * V + key_offsets[:, None] * V + value_offsets[None, :]
        state += tl.load(initial_ptr, mask=state_mask, other=0).to(tl.float32)

        for _ in range(0, sequence_length):
            query = tl.load(q_ptr, mask=key_mask, other=0).to(tl.float32)
            key = tl.load(k_ptr, mask=key_mask, other=0).to(tl.float32)
            value = tl.load(v_ptr, mask=value_mask, other=0).to(tl.float32)

            if USE_QK_L2NORM:
                query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
                key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
            query *= scale

            decay = tl.load(g_ptr).to(tl.float32)
            state *= exp(decay)
            update_scale = tl.load(beta_ptr).to(tl.float32)
            value -= tl.sum(state * key[:, None], axis=0)
            value *= update_scale
            state += key[:, None] * value[None, :]
            result = tl.sum(state * query[:, None], axis=0)
            tl.store(output_ptr, result.to(output_ptr.dtype.element_ty), mask=value_mask)

            q_ptr += stride_q_token
            k_ptr += stride_k_token
            v_ptr += stride_v_token
            g_ptr += stride_g_token
            beta_ptr += stride_beta_token
            output_ptr += stride_output_token

        tl.store(initial_ptr, state.to(initial_ptr.dtype.element_ty), mask=state_mask)


def _ensure_token_strided(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    assert x.shape[0] == 1, "NPU recurrent prefill expects packed [1, tokens, heads, dim] tensors"
    if x.stride()[-2:] != (x.shape[-1], 1):
        x = x.contiguous()
    return x, x.stride(1)


def fused_recurrent_gated_delta_rule_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    _, _, key_head_num, key_dim = k.shape
    value_head_num, value_dim = v.shape[2], v.shape[3]
    assert value_head_num % key_head_num == 0

    sequence_num = len(cu_seqlens) - 1
    if scale is None:
        scale = key_dim**-0.5

    q, stride_q_token = _ensure_token_strided(q)
    k, stride_k_token = _ensure_token_strided(k)
    v, stride_v_token = _ensure_token_strided(v)
    g = g.contiguous()
    assert g.ndim == 3 and beta.ndim == 3

    optimize = (
        q.shape[1] >= 4096 * sequence_num
        and value_head_num > 1
        and key_dim in (64, 128)
        and value_dim in (64, 128)
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and g.dtype == initial_state.dtype == torch.float32
    )
    if optimize:
        if use_qk_l2norm_in_kernel:
            normalized_q = torch.empty(q.shape, dtype=torch.float32, device=q.device)
            normalized_k = torch.empty(k.shape, dtype=torch.float32, device=k.device)
            _prepare_recurrent_qk_npu[(48,)](
                q,
                k,
                normalized_q,
                normalized_k,
                q.shape[1] * key_head_num,
                H=key_head_num,
                K=key_dim,
                STRIDE_Q=stride_q_token,
                STRIDE_K=stride_k_token,
                BK=key_dim,
                BT=16,
                multibuffer=False,
            )
            q, k = normalized_q, normalized_k
        from .chunk_prefill_npu import chunk_prefill_npu

        return chunk_prefill_npu(q, k, v, g, beta, initial_state, cu_seqlens, state_indices, scale)

    beta = beta.contiguous()
    output = torch.empty_like(v)
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(triton.next_power_of_2(value_dim), 32)
    value_tile_num = triton.cdiv(value_dim, block_value)
    sequence_heads = sequence_num * value_head_num
    vector_cores = triton.runtime.driver.active.utils.get_aivector_core_num()
    blocks = value_tile_num * sequence_heads
    # Retain the original compiled kernel for grids without a partial wave.
    kernel = _fused_recurrent_gated_delta_rule_npu_kernel
    grid = (value_tile_num, sequence_heads)
    extra_args = {}
    if blocks > vector_cores and blocks % vector_cores:
        grid = (vector_cores,)
        kernel = _fused_recurrent_gated_delta_rule_npu_safe_kernel
        extra_args = {"SEQUENCE_HEADS": sequence_heads, "BLOCKS": blocks}

    kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output=output,
        initial_state=initial_state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        H=key_head_num,
        HV=value_head_num,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        stride_q_token=stride_q_token,
        stride_k_token=stride_k_token,
        stride_v_token=stride_v_token,
        stride_g_token=g.stride(1),
        stride_beta_token=beta.stride(1),
        stride_output_token=output.stride(1),
        stride_initial_state=initial_state.stride(0),
        USE_QK_L2NORM=use_qk_l2norm_in_kernel,
        num_warps=4,
        num_stages=1,
        multibuffer=False,
        **extra_args,
    )
    return output
