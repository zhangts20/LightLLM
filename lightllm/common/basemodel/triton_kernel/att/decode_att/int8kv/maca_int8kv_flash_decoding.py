from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _align_block_d(head_dim: int, group_size: int) -> int:
    if group_size <= 0:
        raise ValueError(f"quant_group_size must be > 0, got {group_size}")
    block_d = _next_pow2(head_dim)
    while block_d % group_size != 0:
        block_d *= 2
    return block_d


def _q_rows(q_len: int, gqa: int) -> int:
    return max(16, _next_pow2(q_len * gqa))


def _default_run_config(head_dim: int, page_size: Optional[int] = None) -> dict:
    del head_dim
    return {
        "BLOCK_N": 16,
        "BLOCK_SEQ": page_size if page_size else 256,
        "num_warps": 4,
        "num_stages": 2,
        "scenario": "mla",
    }


def _split_count(batch_size: int, max_kv_len: int, block_seq: int) -> int:
    del batch_size
    needed = max(1, triton.cdiv(max(max_kv_len, 1), block_seq))
    return max(1, min(needed, 512))


@triton.jit
def _fwd_int8kv_decode_stage1(
    Q,
    K,
    K_scale,
    V,
    V_scale,
    Page_table,
    B_seqlen,
    Mid_O,
    Mid_LSE,
    sm_scale,
    stride_q_b,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_ks_t,
    stride_ks_h,
    stride_ks_g,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_vs_t,
    stride_vs_h,
    stride_vs_g,
    stride_pt_b,
    stride_pt_p,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    sliding_left,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_LEN: tl.constexpr,
    Q_ROWS: tl.constexpr,
    GQA: tl.constexpr,
    N_Q: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    n_splits = tl.num_programs(2)

    kv_len = tl.load(B_seqlen + cur_batch)
    n_kv_blocks = tl.cdiv(kv_len, PAGE_SIZE)
    if split_id >= n_kv_blocks:
        return

    offs_d = tl.arange(0, BLOCK_D)
    offs_g = tl.arange(0, NUM_GROUPS)
    d_mask = offs_d < HEAD_DIM
    g_mask = offs_g < (HEAD_DIM // GROUP_SIZE)

    offs_row = tl.arange(0, Q_ROWS)
    q_valid = offs_row < (Q_LEN * GQA)
    q_s = offs_row // GQA
    q_h = offs_row % GQA
    q_head = cur_kv_head * GQA + q_h
    q_head = tl.where(q_valid, q_head, cur_kv_head * GQA)
    q_s = tl.where(q_valid, q_s, 0)

    q = tl.load(
        Q
        + cur_batch * stride_q_b
        + q_s[:, None] * stride_q_s
        + q_head[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=q_valid[:, None] & d_mask[None, :],
        other=0.0,
    )

    acc = tl.zeros([Q_ROWS, BLOCK_D], dtype=tl.float32)
    # Finite sentinel: -inf - (-inf) is NaN on some MetaX compiles (D=128 long seq).
    m_i = tl.zeros([Q_ROWS], dtype=tl.float32) - 1.0e10
    l_i = tl.zeros([Q_ROWS], dtype=tl.float32)
    q_pos = kv_len - Q_LEN + q_s

    for kv_block in range(split_id, n_kv_blocks, n_splits):
        block_start = kv_block * PAGE_SIZE
        block_end = tl.minimum(kv_len, block_start + PAGE_SIZE)
        n_tiles = tl.cdiv(block_end - block_start, BLOCK_N)
        offs_n0 = block_start + tl.arange(0, BLOCK_N)
        block_id = tl.load(Page_table + cur_batch * stride_pt_b + kv_block * stride_pt_p).to(tl.int64)
        page_base = block_id * PAGE_SIZE
        for tile in range(0, n_tiles):
            offs_n = offs_n0 + tile * BLOCK_N
            n_mask = offs_n < block_end
            kv_loc = page_base + (offs_n - block_start)

            k_i8 = tl.load(
                K + kv_loc[None, :] * stride_k_t + cur_kv_head * stride_k_h + offs_d[:, None] * stride_k_d,
                mask=n_mask[None, :] & d_mask[:, None],
                other=0,
            )
            k_s = tl.load(
                K_scale + kv_loc[None, :] * stride_ks_t + cur_kv_head * stride_ks_h + offs_g[:, None] * stride_ks_g,
                mask=n_mask[None, :] & g_mask[:, None],
                other=0.0,
            )
            k_i8 = tl.reshape(k_i8, (NUM_GROUPS, GROUP_SIZE, BLOCK_N))
            k_s = tl.reshape(k_s, (NUM_GROUPS, 1, BLOCK_N))
            k = tl.reshape((k_i8.to(q.dtype) * k_s.to(q.dtype)), (BLOCK_D, BLOCK_N))

            qk = tl.dot(q, k).to(tl.float32) * sm_scale
            vis = n_mask[None, :] & q_valid[:, None]
            if CAUSAL:
                vis = vis & (offs_n[None, :] < (q_pos[:, None] + 1))
            if USE_SLIDING_WINDOW:
                win_lo = tl.maximum(q_pos - sliding_left, 0)
                vis = vis & (offs_n[None, :] >= win_lo[:, None])
            qk = tl.where(vis, qk, -1.0e10)

            v_i8 = tl.load(
                V + kv_loc[:, None] * stride_v_t + cur_kv_head * stride_v_h + offs_d[None, :] * stride_v_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0,
            )
            v_s = tl.load(
                V_scale + kv_loc[:, None] * stride_vs_t + cur_kv_head * stride_vs_h + offs_g[None, :] * stride_vs_g,
                mask=n_mask[:, None] & g_mask[None, :],
                other=0.0,
            )
            v_i8 = tl.reshape(v_i8, (BLOCK_N, NUM_GROUPS, GROUP_SIZE))
            v_s = tl.reshape(v_s, (BLOCK_N, NUM_GROUPS, 1))
            v = tl.reshape((v_i8.to(q.dtype) * v_s.to(q.dtype)), (BLOCK_N, BLOCK_D))

            tile_max = tl.max(qk, axis=1)
            new_max = tl.maximum(m_i, tile_max)
            alpha = tl.exp(m_i - new_max)
            p = tl.where(vis, tl.exp(qk - new_max[:, None]), 0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = new_max

    out_head = q_s * N_Q + q_head
    l_safe = tl.where(l_i == 0, 1.0, l_i)
    lse = tl.where(l_i == 0, -1.0e10, m_i + tl.log(l_i))
    tl.store(
        Mid_O + cur_batch * stride_mid_b + out_head[:, None] * stride_mid_h + split_id * stride_mid_s + offs_d[None, :],
        acc / l_safe[:, None],
        mask=q_valid[:, None] & d_mask[None, :],
    )
    tl.store(
        Mid_LSE + cur_batch * stride_lse_b + out_head * stride_lse_h + split_id * stride_lse_s,
        lse,
        mask=q_valid,
    )


@triton.jit
def _fwd_int8kv_decode_stage2(
    Mid_O,
    Mid_LSE,
    B_seqlen,
    O,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_o_b,
    stride_o_s,
    stride_o_h,
    stride_o_d,
    n_splits,
    N_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_out_head = tl.program_id(1)
    kv_len = tl.load(B_seqlen + cur_batch)
    used = tl.minimum(tl.cdiv(kv_len, PAGE_SIZE), n_splits)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    m_i = -1.0e10
    l_i = 0.0

    for split in range(0, used):
        tv = tl.load(
            Mid_O + cur_batch * stride_mid_b + cur_out_head * stride_mid_h + split * stride_mid_s + offs_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        tlogic = tl.load(Mid_LSE + cur_batch * stride_lse_b + cur_out_head * stride_lse_h + split * stride_lse_s)
        new_max = tl.maximum(m_i, tlogic)
        alpha = tl.exp(m_i - new_max)
        beta = tl.exp(tlogic - new_max)
        acc = acc * alpha + beta * tv
        l_i = l_i * alpha + beta
        m_i = new_max

    q_s = cur_out_head // N_Q
    q_h = cur_out_head % N_Q
    l_safe = tl.where(l_i == 0, 1.0, l_i)
    tl.store(
        O + cur_batch * stride_o_b + q_s * stride_o_s + q_h * stride_o_h + offs_d * stride_o_d,
        acc / l_safe,
        mask=d_mask,
    )


@torch.no_grad()
def int8kv_flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    page_table: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    sm_scale: Optional[float] = None,
    causal: bool = True,
    sliding_window: tuple[int, int] = (-1, -1),
    quant_group_size: int = 8,
    max_kv_len: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    alloc_func=torch.empty,
    run_config: Optional[dict] = None,
) -> torch.Tensor:
    if q.dim() != 4:
        raise ValueError(f"q must be (B, Q_LEN, H_Q, D), got {tuple(q.shape)}")
    if page_table is None or page_size is None or page_size <= 0:
        raise ValueError("page_table and page_size are required")
    batch, q_len, n_q, head_dim = q.shape
    n_kv = k.shape[1]
    if n_q % n_kv != 0:
        raise ValueError(f"H_Q={n_q} must be divisible by H_KV={n_kv}")
    if head_dim % quant_group_size != 0:
        raise ValueError(f"head_dim {head_dim} is not divisible by group {quant_group_size}")
    gqa = n_q // n_kv
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    sliding_left = int(sliding_window[0])
    use_swa = sliding_left >= 0
    if use_swa and sliding_window[1] not in (0, -1):
        raise NotImplementedError("right sliding window is not supported")

    cfg = dict(_default_run_config(head_dim, page_size))
    if run_config:
        cfg.update(run_config)
    # Split unit is one page; the unaligned page-walk path is gone.
    block_seq = page_size
    block_n = int(cfg["BLOCK_N"])
    while block_n > 0 and block_seq % block_n != 0:
        block_n //= 2
    if block_n <= 0:
        raise ValueError(f"page_size={page_size} is not compatible with BLOCK_N")

    if max_kv_len is None:
        max_kv_len = int(cache_seqlens.max().item())
    n_splits = int(cfg["n_splits"]) if cfg.get("n_splits") else _split_count(batch, max_kv_len, block_seq)
    n_splits = max(1, n_splits)

    block_d = _align_block_d(head_dim, quant_group_size)
    num_groups = block_d // quant_group_size
    q_rows = _q_rows(q_len, gqa)
    n_out_head = q_len * n_q

    mid_o = alloc_func((batch, n_out_head, n_splits, block_d), dtype=q.dtype, device=q.device)
    mid_lse = alloc_func((batch, n_out_head, n_splits), dtype=torch.float32, device=q.device)
    if out is None:
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)

    # q_len=1 decode already sees the full prefix including the new token.
    use_causal = bool(causal) and q_len > 1

    grid = (batch, n_kv, n_splits)
    _fwd_int8kv_decode_stage1[grid](
        q,
        k,
        k_scale,
        v,
        v_scale,
        page_table,
        cache_seqlens,
        mid_o,
        mid_lse,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        page_table.stride(0),
        page_table.stride(1),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        sliding_left,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        Q_LEN=q_len,
        Q_ROWS=q_rows,
        GQA=gqa,
        N_Q=n_q,
        GROUP_SIZE=quant_group_size,
        NUM_GROUPS=num_groups,
        PAGE_SIZE=page_size,
        CAUSAL=use_causal,
        USE_SLIDING_WINDOW=use_swa,
        num_warps=int(cfg["num_warps"]),
        num_stages=int(cfg["num_stages"]),
        scenario=cfg["scenario"],
    )
    _fwd_int8kv_decode_stage2[(batch, n_out_head)](
        mid_o,
        mid_lse,
        cache_seqlens,
        out,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        n_splits,
        N_Q=n_q,
        BLOCK_D=block_d,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        num_warps=4,
        num_stages=1,
    )
    return out
