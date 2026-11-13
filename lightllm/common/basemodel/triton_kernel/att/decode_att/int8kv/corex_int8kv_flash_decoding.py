from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Optional

from lightllm.common.basemodel.infer_struct import InferStateInfo

_BLOCK_SEQ = 256
# N=16 QK dots stay near 12 TFLOP/s on MR-V100. N=64 reaches the ~48 TFLOP/s
# plateau in isolation, but in the full slim pipeline N=32 wins 14 of 20 shapes
# (rows 12..48 x kv 512..8192) and never loses more than 1%: the smaller tile
# keeps the split QK / PV working sets resident. 32 divides every page size
# the slim gate accepts (they are multiples of _BLOCK_N).
_BLOCK_N = 32
_DOT_MIN = 16
# Slim QK/PV reuses each K/V tile across the verify rows. CUDA graph replay on
# MR-V100 (MTP width 4, head_dim 256, bf16, int8 group 8), milliseconds for 16
# full-attention layers, fused/slim:
#
#   rows  kv=4096  kv=16384  kv=44000
#      4    1.9/1.2    7.0/3.0   18.0/7.3
#      8                        34.9/10.1
#     16                        68.4/18.0
#
# Four rows is one request at mtp_step=3. Slim is already faster there at a few
# thousand tokens and about 2.5x at the long contexts this server actually
# decodes. The launch cannot look at the real KV length: graph capture stores
# graph_max_len_in_batch, and each program stops on b_seq_len.
_SLIM_MIN_BATCH = 4
# _softmax_inplace walks the score row twice (max/sum, then rescale). It is
# bandwidth-bound and one warp per (batch, head) already saturates it; more
# warps only add a cross-warp reduction. Measured at rows=48, kv=4096:
# BLOCK=256/warps=4 (the original) 0.0436ms, BLOCK=512/warps=1 0.0315ms.
_SOFTMAX_BLOCK = 512
_SOFTMAX_WARPS = 1

# The slim kernels hardcode the int8 quantization group and the tile width the
# 256-wide head is split into. They are compile-time constants, not per-call
# values, so they are hoisted out of the launch sites where they used to be
# spelled as bare literals.
_KV_GROUP = 8  # int8 KV quant group; matches llm_kv_quant_group_size
_BLOCK_D = 64  # head_dim slice per _pv_* program; 4 chunks cover head_dim=256
_GROUPS_D = _BLOCK_D // _KV_GROUP
# _qk_* walks head 256 as four chunks of _QK_CHUNK_GROUPS groups. The slim gate
# below requires head_dim == 256, which is what makes that unroll consistent.
_QK_CHUNK_GROUPS = 8
# One program owns every n_splits-th KV tile: Q is loaded once per split, and
# PV is split the same way instead of one CTA walking the whole sequence.
# Cap 64 was the fastest on MR-V100 for the Qwen3.6 TP4 shape (10 requests,
# verify width 4) at 16K and 32K. The grid is min(cap, tiles of max_kv). Under
# CUDA graph max_kv is graph_max_len_in_batch, so the captured launch stays at
# the cap; each program still stops at that request's real length.
_SPLIT_CAP = 64


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _q_rows(gqa: int) -> int:
    return max(_DOT_MIN, _next_pow2(gqa))


def _split_count(batch_size: int, max_kv_len: int) -> int:
    need = max(1, triton.cdiv(max(max_kv_len, 1), _BLOCK_SEQ))
    if batch_size <= 16:
        cap = 128
    elif batch_size <= 64:
        cap = 64
    else:
        cap = 32
    return min(need, cap)


@triton.jit
def _attend_tile(
    q,
    K,
    K_scale,
    V,
    V_scale,
    tok,
    n_mask,
    acc,
    m_i,
    l_i,
    sm_scale,
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
    cur_kv_head,
    q_valid,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    # Head dim is contiguous. Load (token, dim), then transpose K for the dot.
    offs_d = tl.arange(0, HEAD_DIM)
    offs_g = tl.arange(0, NUM_GROUPS)
    if USE_MASK:
        k_i8 = tl.load(
            K + tok[:, None] * stride_k_t + cur_kv_head * stride_k_h + offs_d[None, :] * stride_k_d,
            mask=n_mask[:, None],
            other=0,
        )
        k_s = tl.load(
            K_scale + tok[:, None] * stride_ks_t + cur_kv_head * stride_ks_h + offs_g[None, :] * stride_ks_g,
            mask=n_mask[:, None],
            other=0.0,
        )
    else:
        k_i8 = tl.load(K + tok[:, None] * stride_k_t + cur_kv_head * stride_k_h + offs_d[None, :] * stride_k_d)
        k_s = tl.load(K_scale + tok[:, None] * stride_ks_t + cur_kv_head * stride_ks_h + offs_g[None, :] * stride_ks_g)
    k = tl.reshape(k_i8.to(q.dtype), (BLOCK_N, NUM_GROUPS, GROUP_SIZE)) * tl.reshape(
        k_s.to(q.dtype), (BLOCK_N, NUM_GROUPS, 1)
    )
    k = tl.trans(tl.reshape(k, (BLOCK_N, HEAD_DIM)))
    qk = tl.dot(q, k).to(tl.float32) * sm_scale
    if USE_MASK:
        vis = n_mask[None, :] & q_valid[:, None]
    else:
        vis = q_valid[:, None]
    qk = tl.where(vis, qk, -1.0e10)
    tile_max = tl.max(qk, axis=1)
    new_max = tl.maximum(m_i, tile_max)
    alpha = tl.exp(m_i - new_max)
    p = tl.where(vis, tl.exp(qk - new_max[:, None]), 0.0)
    acc = acc * alpha[:, None]
    if USE_MASK:
        v_i8 = tl.load(
            V + tok[:, None] * stride_v_t + cur_kv_head * stride_v_h + offs_d[None, :] * stride_v_d,
            mask=n_mask[:, None],
            other=0,
        )
        v_s = tl.load(
            V_scale + tok[:, None] * stride_vs_t + cur_kv_head * stride_vs_h + offs_g[None, :] * stride_vs_g,
            mask=n_mask[:, None],
            other=0.0,
        )
    else:
        v_i8 = tl.load(V + tok[:, None] * stride_v_t + cur_kv_head * stride_v_h + offs_d[None, :] * stride_v_d)
        v_s = tl.load(V_scale + tok[:, None] * stride_vs_t + cur_kv_head * stride_vs_h + offs_g[None, :] * stride_vs_g)
    v = tl.reshape(v_i8.to(q.dtype), (BLOCK_N, NUM_GROUPS, GROUP_SIZE)) * tl.reshape(
        v_s.to(q.dtype), (BLOCK_N, NUM_GROUPS, 1)
    )
    v = tl.reshape(v, (BLOCK_N, HEAD_DIM))
    acc += tl.dot(p.to(q.dtype), v).to(tl.float32)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return acc, new_max, l_i


@triton.jit
def _fwd_stage1(
    Q,
    K,
    K_scale,
    V,
    V_scale,
    Req_to_tokens,
    B_req_idx,
    B_seq_len,
    Mid_O,
    Mid_LSE,
    sm_scale,
    stride_q_b,
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
    stride_req_b,
    stride_req_s,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_mid_d,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    GQA: tl.constexpr,
    Q_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    split = tl.program_id(2)
    n_splits = tl.num_programs(2)

    kv_len = tl.load(B_seq_len + cur_batch)
    n_blocks = tl.cdiv(kv_len, BLOCK_SEQ)
    if split >= n_blocks:
        return

    offs_row = tl.arange(0, Q_ROWS)
    q_valid = offs_row < GQA
    q_head = cur_kv_head * GQA + offs_row
    q_head = tl.where(q_valid, q_head, 0)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + cur_batch * stride_q_b + q_head[:, None] * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=q_valid[:, None],
        other=0.0,
    )

    acc = tl.zeros([Q_ROWS, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([Q_ROWS], dtype=tl.float32) - 1.0e10
    l_i = tl.zeros([Q_ROWS], dtype=tl.float32)
    cur_req = tl.load(B_req_idx + cur_batch).to(tl.int64)
    n_full = kv_len // BLOCK_SEQ

    for block in range(split, n_full, n_splits):
        block_start = block * BLOCK_SEQ
        for tile in tl.static_range(BLOCK_SEQ // BLOCK_N):
            tile_start = block_start + tile * BLOCK_N
            offs_n = tile_start + tl.arange(0, BLOCK_N)
            tok = tl.load(Req_to_tokens + cur_req * stride_req_b + offs_n * stride_req_s).to(tl.int64)
            acc, m_i, l_i = _attend_tile(
                q, K, K_scale, V, V_scale, tok, q_valid, acc, m_i, l_i, sm_scale,
                stride_k_t, stride_k_h, stride_k_d, stride_ks_t, stride_ks_h, stride_ks_g,
                stride_v_t, stride_v_h, stride_v_d, stride_vs_t, stride_vs_h, stride_vs_g,
                cur_kv_head, q_valid,
                HEAD_DIM=HEAD_DIM, BLOCK_N=BLOCK_N, GROUP_SIZE=GROUP_SIZE, NUM_GROUPS=NUM_GROUPS, USE_MASK=False,
            )

    tail = n_full * BLOCK_SEQ
    if tail < kv_len and n_full % n_splits == split:
        remain = kv_len - tail
        for tile in range(tl.cdiv(remain, BLOCK_N)):
            tile_start = tail + tile * BLOCK_N
            offs_n = tile_start + tl.arange(0, BLOCK_N)
            n_mask = offs_n < kv_len
            tok = tl.load(
                Req_to_tokens + cur_req * stride_req_b + offs_n * stride_req_s,
                mask=n_mask,
                other=0,
            ).to(tl.int64)
            acc, m_i, l_i = _attend_tile(
                q, K, K_scale, V, V_scale, tok, n_mask, acc, m_i, l_i, sm_scale,
                stride_k_t, stride_k_h, stride_k_d, stride_ks_t, stride_ks_h, stride_ks_g,
                stride_v_t, stride_v_h, stride_v_d, stride_vs_t, stride_vs_h, stride_vs_g,
                cur_kv_head, q_valid,
                HEAD_DIM=HEAD_DIM, BLOCK_N=BLOCK_N, GROUP_SIZE=GROUP_SIZE, NUM_GROUPS=NUM_GROUPS, USE_MASK=True,
            )

    l_safe = tl.where(l_i == 0, 1.0, l_i)
    lse = tl.where(l_i == 0, -1.0e10, m_i + tl.log(l_i))
    tl.store(
        Mid_O
        + cur_batch * stride_mid_b
        + q_head[:, None] * stride_mid_h
        + split * stride_mid_s
        + offs_d[None, :] * stride_mid_d,
        acc / l_safe[:, None],
        mask=q_valid[:, None],
    )
    tl.store(
        Mid_LSE + cur_batch * stride_lse_b + q_head * stride_lse_h + split * stride_lse_s,
        lse,
        mask=q_valid,
    )


@triton.jit
def _fwd_stage2(
    Mid_O,
    Mid_LSE,
    B_seq_len,
    O,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_mid_d,
    stride_lse_b,
    stride_lse_h,
    stride_lse_s,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    n_splits,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    kv_len = tl.load(B_seq_len + cur_batch)
    used = tl.minimum(tl.cdiv(kv_len, BLOCK_SEQ), n_splits)

    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    m_i = -1.0e10
    l_i = 0.0
    offs_o = cur_batch * stride_mid_b + cur_head * stride_mid_h + offs_d * stride_mid_d
    offs_lse = cur_batch * stride_lse_b + cur_head * stride_lse_h
    for split in range(0, used):
        tv = tl.load(Mid_O + offs_o + split * stride_mid_s).to(tl.float32)
        tlogic = tl.load(Mid_LSE + offs_lse + split * stride_lse_s)
        new_max = tl.maximum(m_i, tlogic)
        alpha = tl.exp(m_i - new_max)
        beta = tl.exp(tlogic - new_max)
        acc = acc * alpha + beta * tv
        l_i = l_i * alpha + beta
        m_i = new_max

    l_safe = tl.where(l_i == 0, 1.0, l_i)
    tl.store(O + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d * stride_o_d, acc / l_safe)


@triton.jit
def _softmax_inplace(Scores, B_seq, stride_b, stride_h, stride_t, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    seqlen = tl.load(B_seq + batch)
    base = Scores + batch * stride_b + head * stride_h
    m_i = -1.0e30
    l_i = 0.0
    n_tiles = tl.cdiv(seqlen, BLOCK)
    for tile in range(n_tiles):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < seqlen
        s = tl.load(base + offs * stride_t, mask=mask, other=-1.0e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(s - m_new), axis=0)
        m_i = m_new
    for tile in range(n_tiles):
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offs < seqlen
        s = tl.load(base + offs * stride_t, mask=mask, other=-1.0e30)
        tl.store(base + offs * stride_t, tl.exp(s - m_i) / l_i, mask=mask)


@triton.jit
def _dequant_k_chunk(
    K, KS, tok, n_mask, kv_head, d0, g0,
    stride_kt, stride_kh, stride_kd, stride_kst, stride_ksh, stride_ksg,
    BLOCK_N: tl.constexpr, GROUP: tl.constexpr, CHUNK: tl.constexpr,
):
    offs_cd = tl.arange(0, CHUNK * GROUP)
    offs_cg = tl.arange(0, CHUNK)
    k_i8 = tl.load(
        K + tok[:, None] * stride_kt + kv_head * stride_kh + (d0 + offs_cd)[None, :] * stride_kd,
        mask=n_mask[:, None],
        other=0,
    )
    k_s = tl.load(
        KS + tok[:, None] * stride_kst + kv_head * stride_ksh + (g0 + offs_cg)[None, :] * stride_ksg,
        mask=n_mask[:, None],
        other=0.0,
    )
    return tl.reshape(
        tl.reshape(k_i8.to(tl.bfloat16), (BLOCK_N, CHUNK, GROUP))
        * tl.reshape(k_s.to(tl.bfloat16), (BLOCK_N, CHUNK, 1)),
        (BLOCK_N, CHUNK * GROUP),
    )


@triton.jit
def _qk_grouped(
    Q, K, KS, Req, B_req, B_seq, Scores,
    stride_qb, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_kst, stride_ksh, stride_ksg,
    stride_rb, stride_rs,
    stride_sb, stride_sh, stride_st,
    sm_scale,
    GQA: tl.constexpr,
    Q_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    GROUP: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Head 256 is four CHUNK*GROUP slices. Q is loaded once and reused across
    # the tiles this split owns.
    req = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    n_splits = tl.num_programs(2)
    row0 = req * W

    w_idx = tl.arange(0, W_PAD)
    lens = tl.load(B_seq + row0 + w_idx, mask=w_idx < W, other=0)
    seqlen = tl.max(lens, axis=0)
    n_tiles = tl.cdiv(seqlen, BLOCK_N)
    if split >= n_tiles:
        return

    req_i = tl.load(B_req + row0).to(tl.int64)
    offs_m = tl.arange(0, W_PAD * Q_ROWS)
    t_idx = offs_m // Q_ROWS
    h_idx = offs_m % Q_ROWS
    m_valid = (t_idx < W) & (h_idx < GQA)
    q_head = tl.where(h_idx < GQA, kv_head * GQA + h_idx, 0)
    row = row0 + t_idx
    row_lens = tl.load(B_seq + row0 + t_idx, mask=t_idx < W, other=0)
    offs_cd = tl.arange(0, CHUNK * GROUP)
    q_base = row[:, None] * stride_qb + q_head[:, None] * stride_qh

    q0 = tl.load(Q + q_base + offs_cd[None, :] * stride_qd, mask=m_valid[:, None], other=0.0)
    q1 = tl.load(
        Q + q_base + (CHUNK * GROUP + offs_cd)[None, :] * stride_qd, mask=m_valid[:, None], other=0.0
    )
    q2 = tl.load(
        Q + q_base + (2 * CHUNK * GROUP + offs_cd)[None, :] * stride_qd, mask=m_valid[:, None], other=0.0
    )
    q3 = tl.load(
        Q + q_base + (3 * CHUNK * GROUP + offs_cd)[None, :] * stride_qd, mask=m_valid[:, None], other=0.0
    )

    offs_n = tl.arange(0, BLOCK_N)
    for tile in range(split, n_tiles, n_splits):
        start = tile * BLOCK_N
        pos = start + offs_n
        n_mask = pos < seqlen
        loc0 = tl.load(Req + req_i * stride_rb + start * stride_rs).to(tl.int64)
        tok = loc0 + offs_n
        score = tl.zeros((W_PAD * Q_ROWS, BLOCK_N), dtype=tl.float32)
        score += tl.dot(q0, tl.trans(_dequant_k_chunk(
            K, KS, tok, n_mask, kv_head, 0, 0,
            stride_kt, stride_kh, stride_kd, stride_kst, stride_ksh, stride_ksg,
            BLOCK_N=BLOCK_N, GROUP=GROUP, CHUNK=CHUNK,
        )))
        score += tl.dot(q1, tl.trans(_dequant_k_chunk(
            K, KS, tok, n_mask, kv_head, CHUNK * GROUP, CHUNK,
            stride_kt, stride_kh, stride_kd, stride_kst, stride_ksh, stride_ksg,
            BLOCK_N=BLOCK_N, GROUP=GROUP, CHUNK=CHUNK,
        )))
        score += tl.dot(q2, tl.trans(_dequant_k_chunk(
            K, KS, tok, n_mask, kv_head, 2 * CHUNK * GROUP, 2 * CHUNK,
            stride_kt, stride_kh, stride_kd, stride_kst, stride_ksh, stride_ksg,
            BLOCK_N=BLOCK_N, GROUP=GROUP, CHUNK=CHUNK,
        )))
        score += tl.dot(q3, tl.trans(_dequant_k_chunk(
            K, KS, tok, n_mask, kv_head, 3 * CHUNK * GROUP, 3 * CHUNK,
            stride_kt, stride_kh, stride_kd, stride_kst, stride_ksh, stride_ksg,
            BLOCK_N=BLOCK_N, GROUP=GROUP, CHUNK=CHUNK,
        )))
        tl.store(
            Scores + row[:, None] * stride_sb + q_head[:, None] * stride_sh + pos[None, :] * stride_st,
            score * sm_scale,
            mask=m_valid[:, None] & (pos[None, :] < row_lens[:, None]),
        )


@triton.jit
def _pv_grouped(
    Probs, V, VS, Req, B_req, B_seq, Mid,
    stride_pb, stride_ph, stride_pt,
    stride_vt, stride_vh, stride_vd,
    stride_vst, stride_vsh, stride_vsg,
    stride_rb, stride_rs,
    stride_mb, stride_mh, stride_ms, stride_md,
    GQA: tl.constexpr,
    Q_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    GROUP: tl.constexpr,
    GROUPS_D: tl.constexpr,
    DCHUNKS: tl.constexpr,
):
    req = tl.program_id(0)
    kv_head = tl.program_id(1)
    pid = tl.program_id(2)
    split = pid // DCHUNKS
    chunk = pid - split * DCHUNKS
    n_splits = tl.num_programs(2) // DCHUNKS
    row0 = req * W

    w_idx = tl.arange(0, W_PAD)
    lens = tl.load(B_seq + row0 + w_idx, mask=w_idx < W, other=0)
    seqlen = tl.max(lens, axis=0)
    n_tiles = tl.cdiv(seqlen, BLOCK_N)
    req_i = tl.load(B_req + row0).to(tl.int64)

    offs_m = tl.arange(0, W_PAD * Q_ROWS)
    t_idx = offs_m // Q_ROWS
    h_idx = offs_m % Q_ROWS
    m_valid = (t_idx < W) & (h_idx < GQA)
    q_head = tl.where(h_idx < GQA, kv_head * GQA + h_idx, 0)
    row = row0 + t_idx
    row_lens = tl.load(B_seq + row0 + t_idx, mask=t_idx < W, other=0)

    d0 = chunk * BLOCK_D
    g0 = chunk * GROUPS_D
    offs_d = tl.arange(0, BLOCK_D)
    offs_g = tl.arange(0, GROUPS_D)
    acc = tl.zeros((W_PAD * Q_ROWS, BLOCK_D), dtype=tl.float32)
    if split < n_tiles:
        offs_n = tl.arange(0, BLOCK_N)
        for tile in range(split, n_tiles, n_splits):
            start = tile * BLOCK_N
            pos = start + offs_n
            n_mask = pos < seqlen
            loc0 = tl.load(Req + req_i * stride_rb + start * stride_rs).to(tl.int64)
            tok = loc0 + offs_n
            prob = tl.load(
                Probs + row[:, None] * stride_pb + q_head[:, None] * stride_ph + pos[None, :] * stride_pt,
                mask=m_valid[:, None] & (pos[None, :] < row_lens[:, None]),
                other=0.0,
            )
            v_i8 = tl.load(
                V + tok[:, None] * stride_vt + kv_head * stride_vh + (d0 + offs_d)[None, :] * stride_vd,
                mask=n_mask[:, None],
                other=0,
            )
            v_s = tl.load(
                VS + tok[:, None] * stride_vst + kv_head * stride_vsh + (g0 + offs_g)[None, :] * stride_vsg,
                mask=n_mask[:, None],
                other=0.0,
            )
            v_bf = tl.reshape(
                tl.reshape(v_i8.to(tl.bfloat16), (BLOCK_N, GROUPS_D, GROUP))
                * tl.reshape(v_s.to(tl.bfloat16), (BLOCK_N, GROUPS_D, 1)),
                (BLOCK_N, BLOCK_D),
            )
            acc += tl.dot(prob.to(tl.bfloat16), v_bf)
    tl.store(
        Mid + row[:, None] * stride_mb + q_head[:, None] * stride_mh + split * stride_ms
        + (d0 + offs_d)[None, :] * stride_md,
        acc,
        mask=m_valid[:, None],
    )


@triton.jit
def _pv_reduce(
    Mid, Out, B_seq,
    stride_mb, stride_mh, stride_ms, stride_md,
    stride_ob, stride_oh, stride_od,
    n_splits,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    seqlen = tl.load(B_seq + row)
    offs = tl.arange(0, HEAD_DIM)
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for split in range(n_splits):
        if split * BLOCK_N < seqlen:
            acc += tl.load(
                Mid + row * stride_mb + head * stride_mh + split * stride_ms + offs * stride_md
            )
    tl.store(Out + row * stride_ob + head * stride_oh + offs * stride_od, acc.to(tl.bfloat16))


def _decode_grouped(
    q, infer_state, cache_k, cache_k_scale, cache_v, cache_v_scale, out, alloc_tensor_func,
    batch_size, q_head_num, kv_head_num, gqa, q_rows, verify_width,
):
    max_kv = max(int(infer_state.max_kv_seq_len), 1)
    score_len = triton.cdiv(max_kv, _BLOCK_N) * _BLOCK_N
    tiles = score_len // _BLOCK_N
    n_splits = min(_SPLIT_CAP, tiles)
    head_dim = q.shape[-1]
    dchunks = head_dim // _BLOCK_D
    scores = alloc_tensor_func((batch_size, q_head_num, score_len), torch.float32, q.device)
    partial = alloc_tensor_func((batch_size, q_head_num, n_splits, head_dim), torch.float32, q.device)
    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out
    sm_scale = head_dim ** -0.5
    n_req = batch_size // verify_width
    w_pad = triton.next_power_of_2(verify_width)
    tile_rows = _next_pow2(gqa)
    if tile_rows < q_rows and tile_rows * w_pad >= _DOT_MIN:
        q_rows = tile_rows
    table = infer_state.req_manager.req_to_token_indexs
    _qk_grouped[(n_req, kv_head_num, n_splits)](
        q, cache_k, cache_k_scale,
        table, infer_state.b_req_idx, infer_state.b_seq_len, scores,
        q.stride(0), q.stride(1), q.stride(2),
        cache_k.stride(0), cache_k.stride(1), cache_k.stride(2),
        cache_k_scale.stride(0), cache_k_scale.stride(1), cache_k_scale.stride(2),
        table.stride(0), table.stride(1),
        scores.stride(0), scores.stride(1), scores.stride(2),
        sm_scale,
        GQA=gqa, Q_ROWS=q_rows, BLOCK_N=_BLOCK_N,
        W=verify_width, W_PAD=w_pad,
        GROUP=_KV_GROUP, CHUNK=_QK_CHUNK_GROUPS,
        num_warps=4, num_stages=1,
    )
    _softmax_inplace[(batch_size, q_head_num)](
        scores, infer_state.b_seq_len,
        scores.stride(0), scores.stride(1), scores.stride(2),
        BLOCK=_SOFTMAX_BLOCK, num_warps=_SOFTMAX_WARPS, num_stages=1,
    )
    o_view = o_tensor.view(batch_size, q_head_num, head_dim)
    _pv_grouped[(n_req, kv_head_num, n_splits * dchunks)](
        scores, cache_v, cache_v_scale,
        table, infer_state.b_req_idx, infer_state.b_seq_len, partial,
        scores.stride(0), scores.stride(1), scores.stride(2),
        cache_v.stride(0), cache_v.stride(1), cache_v.stride(2),
        cache_v_scale.stride(0), cache_v_scale.stride(1), cache_v_scale.stride(2),
        table.stride(0), table.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2), partial.stride(3),
        GQA=gqa, Q_ROWS=q_rows, BLOCK_N=_BLOCK_N, BLOCK_D=_BLOCK_D,
        W=verify_width, W_PAD=w_pad,
        GROUP=_KV_GROUP, GROUPS_D=_GROUPS_D, DCHUNKS=dchunks,
        num_warps=4, num_stages=1,
    )
    _pv_reduce[(batch_size, q_head_num)](
        partial, o_view, infer_state.b_seq_len,
        partial.stride(0), partial.stride(1), partial.stride(2), partial.stride(3),
        o_view.stride(0), o_view.stride(1), o_view.stride(2),
        n_splits, BLOCK_N=_BLOCK_N, HEAD_DIM=head_dim,
        num_warps=4, num_stages=1,
    )
    return o_tensor


@torch.no_grad()
def token_decode_attention_flash_decoding(
    q: torch.Tensor,
    infer_state: InferStateInfo,
    cache_k: torch.Tensor,
    cache_k_scale: torch.Tensor,
    cache_v: torch.Tensor,
    cache_v_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    alloc_tensor_func=torch.empty,
    n_splits: Optional[int] = None,
    verify_width: int = 1,
) -> torch.Tensor:
    batch_size = infer_state.batch_size
    q_head_num = q.shape[1]
    head_dim = q.shape[2]
    kv_head_num = cache_k.shape[1]
    if q_head_num % kv_head_num != 0:
        raise ValueError(f"q heads {q_head_num} must be divisible by kv heads {kv_head_num}")

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q dtype must be fp16 or bf16, got {q.dtype}")

    if head_dim not in (64, 128, 256):
        raise ValueError(f"head dim {head_dim} is not supported")

    group_size = head_dim // cache_k_scale.shape[-1]
    num_groups = head_dim // group_size
    if _next_pow2(num_groups) != num_groups:
        raise ValueError(f"head dim / group = {num_groups} must be a power of 2")

    gqa = q_head_num // kv_head_num
    q_rows = _q_rows(gqa)
    q = q.view(batch_size, q_head_num, head_dim)
    if (
        head_dim == 256
        and group_size == 8
        and q.dtype == torch.bfloat16
        and q_rows == 16
        and batch_size >= _SLIM_MIN_BATCH
    ):
        from lightllm.utils.envs_utils import get_page_size

        if get_page_size() % _BLOCK_N == 0:
            return _decode_grouped(
                q, infer_state, cache_k, cache_k_scale, cache_v, cache_v_scale,
                out, alloc_tensor_func, batch_size, q_head_num, kv_head_num, gqa, q_rows,
                verify_width,
            )
    if n_splits is None:
        n_splits = _split_count(batch_size, int(infer_state.max_kv_seq_len))

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out
    mid_o = alloc_tensor_func((batch_size, q_head_num, n_splits, head_dim), torch.float32, q.device)
    mid_lse = alloc_tensor_func((batch_size, q_head_num, n_splits), torch.float32, q.device)
    sm_scale = head_dim ** -0.5

    _fwd_stage1[(batch_size, kv_head_num, n_splits)](
        q,
        cache_k,
        cache_k_scale,
        cache_v,
        cache_v_scale,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        mid_o,
        mid_lse,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cache_k.stride(0),
        cache_k.stride(1),
        cache_k.stride(2),
        cache_k_scale.stride(0),
        cache_k_scale.stride(1),
        cache_k_scale.stride(2),
        cache_v.stride(0),
        cache_v.stride(1),
        cache_v.stride(2),
        cache_v_scale.stride(0),
        cache_v_scale.stride(1),
        cache_v_scale.stride(2),
        infer_state.req_manager.req_to_token_indexs.stride(0),
        infer_state.req_manager.req_to_token_indexs.stride(1),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        GQA=gqa,
        Q_ROWS=q_rows,
        HEAD_DIM=head_dim,
        BLOCK_N=_BLOCK_N,
        BLOCK_SEQ=_BLOCK_SEQ,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        num_warps=4,
        num_stages=1,
    )
    _fwd_stage2[(batch_size, q_head_num)](
        mid_o,
        mid_lse,
        infer_state.b_seq_len,
        o_tensor.view(batch_size, q_head_num, head_dim),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        o_tensor.stride(0),
        o_tensor.stride(1),
        o_tensor.stride(2),
        n_splits,
        HEAD_DIM=head_dim,
        BLOCK_SEQ=_BLOCK_SEQ,
        num_warps=4,
        num_stages=1,
    )
    return o_tensor
