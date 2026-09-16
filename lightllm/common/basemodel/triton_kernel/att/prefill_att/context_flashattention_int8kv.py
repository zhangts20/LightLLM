import torch

import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice


@triton.jit
def _fwd_int8kv_kernel(
    Q,
    K,
    V,
    KScale,
    VScale,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    kv_group_num,
    b_prompt_cache_len,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    SLIDING_WINDOW_LEFT: tl.constexpr,
):
    tl.static_assert(PAGE_SIZE % BLOCK_N == 0)
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    q_start = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    q_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    req_idx = tl.load(B_req_idx + cur_batch)

    block_start = BLOCK_M * start_m
    offs_m = block_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_pos = prompt_cache_len + offs_m

    q_offs = (
        (q_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + q_offs, mask=offs_m[:, None] < q_len, other=0.0)
    # Quantize Q per row on chip so QK can use the native INT8 Cube path.
    q_absmax = tl.max(tl.abs(q).to(tl.float32), axis=1)
    q_scale = tl.maximum(q_absmax / 127.0, 1.0e-12)
    q_quant = libdevice.round(q.to(tl.float32) / q_scale[:, None]).to(tl.int8)
    if not USE_SLIDING_WINDOW:
        q_scale *= sm_scale

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    active = tl.where(block_start < q_len, 1, 0)
    kv_end = tl.minimum(block_start + BLOCK_M + prompt_cache_len, q_len + prompt_cache_len)
    if USE_SLIDING_WINDOW:
        kv_start = tl.maximum(block_start + prompt_cache_len - SLIDING_WINDOW_LEFT, 0)
    else:
        kv_start = 0

    for start_n in range(0, active * (kv_end - kv_start), BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_pos = kv_start + start_n + offs_n
        valid_k = k_pos < kv_end
        if not USE_SLIDING_WINDOW:
            # BLOCK_N divides PAGE_SIZE and k_pos is BLOCK_N-aligned. A tile
            # cannot cross a cache-page boundary, so one page-slot lookup is
            # enough and all token rows in the tile are physically contiguous.
            page_first_loc = tl.load(
                Req_to_tokens + req_idx * stride_req_to_tokens_b + (kv_start + start_n) * stride_req_to_tokens_s
            ).to(tl.int64)
            kv_loc = page_first_loc + offs_n
        else:
            kv_loc = tl.load(
                Req_to_tokens + req_idx * stride_req_to_tokens_b + k_pos * stride_req_to_tokens_s,
                mask=valid_k,
                other=0,
            ).to(tl.int64)

        # The no-window path turns kv_loc into a provably contiguous page
        # range. Load [N, D] vectorized and transpose on chip; the discrete
        # row-by-row fallback is only needed for sliding-window gathers.
        if not USE_SLIDING_WINDOW:
            k_row_offs = kv_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
            k_quant = tl.load(K + k_row_offs)
            k_scale = tl.load(KScale + kv_loc)
            k_scale = tl.where(valid_k, k_scale, 0.0)
            k = tl.trans(k_quant)
        else:
            k_rows = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.bfloat16)
            for token_i in range(0, BLOCK_N):
                token_loc = tl.get_element(kv_loc, (token_i,))
                token_valid = tl.get_element(valid_k, (token_i,))
                k_row_offs = token_loc * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
                k_row_q = tl.load(K + k_row_offs, mask=token_valid, other=0.0)
                k_row_scale = tl.load(KScale + token_loc, mask=token_valid, other=0.0)
                k_row = (k_row_q * k_row_scale).to(tl.bfloat16)
                k_rows = tl.insert_slice(k_rows, k_row, (token_i, 0), (1, BLOCK_DMODEL), (1, 1))
            k = tl.trans(k_rows)
        if not USE_SLIDING_WINDOW:
            qk = tl.dot(q_quant, k, out_dtype=tl.int32).to(tl.float32)
            qk *= q_scale[:, None] * k_scale[None, :]
        else:
            qk = tl.dot(q, k)

        if USE_SLIDING_WINDOW:
            causal = (q_pos[:, None] >= k_pos[None, :]) & (
                q_pos[:, None] - k_pos[None, :] <= SLIDING_WINDOW_LEFT
            )
        else:
            causal = q_pos[:, None] >= k_pos[None, :]
        if USE_SLIDING_WINDOW:
            qk = tl.where(causal & valid_k[None, :], qk * sm_scale, -1.0e8)
        else:
            qk = tl.where(causal & valid_k[None, :], qk, -1.0e8)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc *= alpha[:, None]

        v_offs = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v_quant = tl.load(V + v_offs)
        v_scale = tl.load(VScale + kv_loc)
        v_scale = tl.where(valid_k, v_scale, 0.0)
        # P @ (diag(scale) @ Vq) == (P * scale) @ Vq. Scaling the 64x64
        # probability tile is cheaper than broadcasting over 64xD values and
        # avoids materializing a scaled-V FP32 tile before BF16 Cube PV.
        p_v = (p * v_scale[None, :]).to(tl.bfloat16)
        v = v_quant.to(tl.bfloat16)
        acc = tl.dot(p_v, v, acc)
        m_i = m_ij

    acc /= l_i[:, None]
    out_offs = (
        (q_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(Out + out_offs, acc, mask=offs_m[:, None] < q_len)


@torch.no_grad()
def context_attention_fwd_int8kv(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    max_q_len: int,
    req_to_token_indexs: torch.Tensor,
    sliding_window: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    head_dim = q.shape[-1]
    assert head_dim in {16, 32, 64, 128, 256, 512}
    assert q.dtype == torch.bfloat16, "Ascend fused INT8 KV prefill currently supports BF16 query only"
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32

    # Adjacent block/page dimensions form the physical token dimension.
    page_size = k_cache.shape[1]
    k_cache = k_cache.flatten(0, 1)
    v_cache = v_cache.flatten(0, 1)
    k_scale = k_scale.flatten()
    v_scale = v_scale.flatten()

    # Disable compiler multi-buffering below so the scaled K/V tiles are not
    # duplicated in UB. This makes 64x64 fit and remain numerically stable on
    # the validated B2C stack; 128-row or 128-column variants still overflow.
    block_m = 64
    if head_dim >= 512:
        block_m = 16
    elif head_dim >= 256:
        block_m = 16
    block_n = 64

    use_sliding_window = sliding_window != (-1, -1)
    if use_sliding_window:
        assert int(sliding_window[1]) == 0
    sliding_window_left = int(sliding_window[0])

    batch = b_seq_len.shape[0]
    q_heads = q.shape[1]
    kv_heads = k_cache.shape[1]
    assert q_heads % kv_heads == 0
    grid = (triton.cdiv(max_q_len, block_m), batch * q_heads, 1)
    _fwd_int8kv_kernel[grid](
        q,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        head_dim**-0.5 * 1.4426950408889634,
        out,
        b_start_loc,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *out.stride(),
        *req_to_token_indexs.stride(),
        kv_group_num=q_heads // kv_heads,
        b_prompt_cache_len=b_prompt_cache_len,
        H=q_heads,
        BLOCK_DMODEL=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        PAGE_SIZE=page_size,
        USE_SLIDING_WINDOW=use_sliding_window,
        SLIDING_WINDOW_LEFT=sliding_window_left,
        num_warps=4,
        num_stages=2,
        multibuffer=False,
        enable_auto_bind_sub_block=False,
    )
    return out
