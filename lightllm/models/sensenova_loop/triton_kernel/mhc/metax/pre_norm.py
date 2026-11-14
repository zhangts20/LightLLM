# SPDX-License-Identifier: Apache-2.0

"""MetaX mHC pre-norm.

main 的 prepare kernel 用 Python range 展开 Sinkhorn，MetaX 的 make_ttgir 会 segfault。
这里的 kernel 用 tl.static_range。两 stream、tokens <= 128 的投影用 split-K。
这个形状上 split-K 到 128 行仍不慢于 fp32 GEMM，256 行起 GEMM 更快。
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from lightllm.models.sensenova_loop.triton_kernel.mhc.pre_norm import _hc_pre_combine_norm_kernel


def is_maca_mhc() -> bool:
    try:
        from lightllm.utils.envs_utils import get_env_start_args

        return getattr(get_env_start_args(), "hardware_platform", "cuda") == "maca"
    except Exception:
        return "metax" in (getattr(torch, "__version__", "") or "").lower()


@triton.jit
def _sinkhorn_step_metax(logits):
    logits = logits - tl.max(logits, axis=1)[:, None]
    logits = logits - tl.log(tl.sum(tl.exp(logits), axis=1))[:, None]
    logits = logits - tl.max(logits, axis=0)[None, :]
    logits = logits - tl.log(tl.sum(tl.exp(logits), axis=0))[None, :]
    return logits


@triton.jit
def _hc_proj_metax_kernel(
    x,
    fn,
    gemm_partial,
    sqrsum_partial,
    stride_x,
    stride_fn,
    gemm_stride_s,
    gemm_stride_m,
    sqrsum_stride_s,
    K,
    M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    split = tl.program_id(1)
    k_per = tl.cdiv(K, tl.num_programs(1))
    start = split * k_per
    acc = tl.zeros((M,), dtype=tl.float32)
    sq = 0.0
    mix_offsets = tl.arange(0, M)
    for k0 in tl.range(0, k_per, BLOCK_K):
        k_offsets = start + k0 + tl.arange(0, BLOCK_K)
        mask = (k_offsets < start + k_per) & (k_offsets < K)
        values = tl.load(x + token * stride_x + k_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(fn + mix_offsets[:, None] * stride_fn + k_offsets[None, :], mask=mask[None, :], other=0.0)
        acc += tl.sum(weight * values[None, :], axis=1)
        sq += tl.sum(values * values, axis=0)
    tl.store(gemm_partial + split * gemm_stride_s + token * gemm_stride_m + mix_offsets, acc)
    tl.store(sqrsum_partial + split * sqrsum_stride_s + token, sq)


@triton.jit
def _hc_prepare_prenorm_metax_kernel(
    gemm_partial,
    sqrsum_partial,
    base,
    pre,
    post,
    residual_mix,
    gemm_stride_s,
    gemm_stride_m: tl.constexpr,
    sqrsum_stride_s,
    sqrsum_stride_m: tl.constexpr,
    pre_stride_m: tl.constexpr,
    residual_stride_m: tl.constexpr,
    FLATTENED_HIDDEN: tl.constexpr,
    RMS_EPS: tl.constexpr,
    STREAMS: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    N_SPLITS: tl.constexpr,
):
    token = tl.program_id(0)
    stream_offsets = tl.arange(0, STREAMS)
    matrix_offsets = tl.arange(0, STREAMS * STREAMS)
    pre_raw = tl.zeros((STREAMS,), dtype=tl.float32)
    post_raw = tl.zeros((STREAMS,), dtype=tl.float32)
    matrix_raw = tl.zeros((STREAMS * STREAMS,), dtype=tl.float32)
    sqrsum = 0.0
    for split in tl.static_range(N_SPLITS):
        partial_base = gemm_partial + split * gemm_stride_s + token * gemm_stride_m
        pre_raw += tl.load(partial_base + stream_offsets)
        post_raw += tl.load(partial_base + STREAMS + stream_offsets)
        matrix_raw += tl.load(partial_base + 2 * STREAMS + matrix_offsets)
        sqrsum += tl.load(sqrsum_partial + split * sqrsum_stride_s + token * sqrsum_stride_m)
    inv_rms = tl.rsqrt(sqrsum / FLATTENED_HIDDEN + RMS_EPS)
    pre_raw *= inv_rms
    post_raw *= inv_rms
    matrix_raw *= inv_rms

    pre_values = tl.sigmoid(pre_raw + tl.load(base + stream_offsets))
    post_values = 2.0 * tl.sigmoid(post_raw + tl.load(base + STREAMS + stream_offsets))

    residual_logits = matrix_raw + tl.load(base + 2 * STREAMS + matrix_offsets)
    residual_logits = tl.reshape(residual_logits, (STREAMS, STREAMS))
    if SINKHORN_ITERS > 0:
        for _ in tl.static_range(SINKHORN_ITERS):
            residual_logits = _sinkhorn_step_metax(residual_logits)
    residual_logits = tl.exp(residual_logits)

    tl.store(pre + token * pre_stride_m + stream_offsets, pre_values)
    tl.store(post + token * pre_stride_m + stream_offsets, post_values)
    tl.store(
        residual_mix + token * residual_stride_m + matrix_offsets,
        tl.reshape(residual_logits, (STREAMS * STREAMS,)),
    )


def hc_pre_norm_metax(
    x: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    streams: int,
    rms_eps: float,
    norm_eps: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.ndim == 2 and x.shape[-1] % streams == 0
    assert streams in (2, 4)
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32) and fn.dtype == torch.float32
    assert x.is_contiguous() and fn.is_contiguous() and norm_weight.is_contiguous()
    assert fn.shape == (2 * streams + streams * streams, x.shape[1])
    tokens, flattened_hidden = x.shape
    hidden = flattened_hidden // streams
    mix_size = fn.shape[0]
    # 正确性和性能分界
    if streams == 2 and tokens <= 128:
        n_splits = 16
        gemm_partial = torch.empty((n_splits, tokens, mix_size), dtype=torch.float32, device=x.device)
        sqrsum_partial = torch.empty((n_splits, tokens), dtype=torch.float32, device=x.device)
        _hc_proj_metax_kernel[(tokens, n_splits)](
            x,
            fn,
            gemm_partial,
            sqrsum_partial,
            x.stride(0),
            fn.stride(0),
            gemm_partial.stride(0),
            gemm_partial.stride(1),
            sqrsum_partial.stride(0),
            flattened_hidden,
            M=mix_size,
            BLOCK_K=128,
            num_warps=4,
        )
    else:
        x_fp32 = x.float()
        gemm_partial = (x_fp32 @ fn.T).unsqueeze(0)
        sqrsum_partial = x_fp32.square().sum(dim=-1).unsqueeze(0)
        n_splits = 1

    pre = torch.empty((tokens, streams), dtype=torch.float32, device=x.device)
    post = torch.empty_like(pre)
    residual_mix = torch.empty((tokens, streams, streams), dtype=torch.float32, device=x.device)
    _hc_prepare_prenorm_metax_kernel[(tokens,)](
        gemm_partial,
        sqrsum_partial,
        base,
        pre,
        post,
        residual_mix,
        gemm_partial.stride(0),
        gemm_partial.stride(1),
        sqrsum_partial.stride(0),
        sqrsum_partial.stride(1),
        pre.stride(0),
        residual_mix.stride(0),
        FLATTENED_HIDDEN=flattened_hidden,
        RMS_EPS=rms_eps,
        STREAMS=streams,
        SINKHORN_ITERS=sinkhorn_iters,
        N_SPLITS=n_splits,
        num_warps=1,
    )

    layer_input = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    block_h = triton.next_power_of_2(hidden)
    _hc_pre_combine_norm_kernel[(tokens,)](
        x,
        pre,
        norm_weight,
        layer_input,
        hidden=hidden,
        x_stride_m=x.stride(0),
        pre_stride_m=pre.stride(0),
        out_stride_m=layer_input.stride(0),
        STREAMS=streams,
        NORM_EPS=norm_eps,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return layer_input, residual_mix, post
