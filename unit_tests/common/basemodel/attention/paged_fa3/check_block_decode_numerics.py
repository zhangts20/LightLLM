"""Standalone CUDA/MACA numerical check; does not load a model or allocate KV pages.

Run from the LightLLM root on a spare visible device:
  PYTHONPATH=. python unit_tests/common/basemodel/attention/paged_fa3/check_block_decode_numerics.py --device 0 --graph

CUDA_VISIBLE_DEVICES (or the MACA equivalent) should be set by the caller.
"""
import argparse
import math

import torch
from flash_attn import flash_attn_varlen_func

from lightllm.common.basemodel.attention.paged_fa3.fp import _gather_block_kv_kernel


def check(page_size, dtype, graph_enabled, device, heads=4, kv_heads=2, dim=64, max_kv_len=None):
    torch.manual_seed(20260917)
    max_len = max_kv_len if max_kv_len is not None else 2 * page_size + 16
    # Two complete five-query blocks plus a short trailing graph HOLD group.
    q_lengths = [5, 5, 2]
    lengths = torch.tensor([page_size + 6, page_size + 10, 1], device=device, dtype=torch.int32)
    cu_q = torch.tensor([0, 5, 10, 12], device=device, dtype=torch.int32)
    cu_k = torch.cat((torch.zeros(1, device=device, dtype=torch.int32), lengths.cumsum(0).int()))
    req_ids = torch.tensor([1, 2, 0], device=device, dtype=torch.int32)
    mapping = torch.full((3, max_len), -1, device=device, dtype=torch.int32)
    mapping[0, 0] = 0
    for req, prefix, scratch in [(1, page_size + 1, 6 * page_size), (2, page_size + 5, 8 * page_size)]:
        mapping[req, :prefix] = torch.arange(2 * page_size, 2 * page_size + prefix, device=device)
        mapping[req, prefix:prefix + 5] = torch.arange(scratch, scratch + 5, device=device)
    cache = torch.randn(12 * page_size, 2 * kv_heads, dim, device=device, dtype=dtype)
    original_cache = cache.clone()
    k, v = cache[:, :kv_heads], cache[:, kv_heads:]
    q = torch.randn(sum(q_lengths), heads, dim, device=device, dtype=dtype)
    packed_k = torch.empty((3 * max_len, kv_heads, dim), device=device, dtype=dtype)
    packed_v = torch.empty_like(packed_k)

    def forward():
        _gather_block_kv_kernel[((max_len * kv_heads * dim + 255) // 256, 3)](
            k, v, packed_k, packed_v, mapping, req_ids, lengths, cu_k,
            mapping.stride(0), k.stride(0), v.stride(0), k.stride(1), v.stride(1),
            k.stride(2), v.stride(2), kv_heads, dim, 256,
        )
        return flash_attn_varlen_func(
            q=q, k=packed_k, v=packed_v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=5, max_seqlen_k=max_len, softmax_scale=1 / math.sqrt(dim),
            causal=False, window_size=(-1, -1), softcap=0.0,
        )

    def validate(output, phase):
        refs, start = [], 0
        for group, (req, q_len) in enumerate(zip(req_ids.tolist(), q_lengths)):
            n = int(lengths[group])
            slots = mapping[req, :n].long()
            key = k[slots].float().repeat_interleave(heads // kv_heads, dim=1)
            val = v[slots].float().repeat_interleave(heads // kv_heads, dim=1)
            query = q[start:start + q_len].float()
            scores = torch.einsum("qhd,khd->hqk", query, key) / math.sqrt(dim)
            refs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), val))
            packed_start = int(cu_k[group])
            torch.testing.assert_close(packed_k[packed_start:packed_start + n], k[slots], atol=0, rtol=0)
            torch.testing.assert_close(packed_v[packed_start:packed_start + n], v[slots], atol=0, rtol=0)
            start += q_len
        reference = torch.cat(refs)
        atol = 0.025 if dtype == torch.bfloat16 else 0.004
        torch.testing.assert_close(output.float(), reference, atol=atol, rtol=atol)
        torch.testing.assert_close(cache, original_cache, atol=0, rtol=0)
        print(f"PASS page_size={page_size} dtype={dtype} heads={heads} kv_heads={kv_heads} head_dim={dim} capacity={max_len} phase={phase} max_error={(output.float() - reference).abs().max().item():.6f}", flush=True)

    validate(forward(), "eager")
    if graph_enabled:
        # Warm compilation and FlashAttention allocations on a side stream.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward()
        graph.replay()
        validate(output, "capture-replay")
        # Change physical mapping, query data, and packed offsets in-place.
        # Reusing only capture-time values would now produce a wrong result.
        lengths[1] -= 2
        mapping[1, :page_size + 6] = mapping[1, :page_size + 6].flip(0)
        cu_k[1:].copy_(lengths.cumsum(0).int())
        q.mul_(0.5)
        graph.replay()
        validate(output, "changed-mapping-replay")
    torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="index within caller-selected visible devices")
    parser.add_argument("--graph", action="store_true", help="also validate capture and changed-data replay")
    parser.add_argument("--page-sizes", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--max-kv-len", type=int, default=None,
                        help="packed capacity per group; use 8192 to check graph-max allocation/grid with short actual KV")
    args = parser.parse_args()
    if args.heads <= 0 or args.kv_heads <= 0 or args.heads % args.kv_heads:
        parser.error("heads and kv-heads must be positive, with heads divisible by kv-heads")
    if args.head_dim <= 0 or any(page <= 0 or page % 16 for page in args.page_sizes):
        parser.error("head-dim must be positive; page sizes must be positive multiples of 16")
    if args.max_kv_len is not None and args.max_kv_len < max(args.page_sizes) + 10:
        parser.error("max-kv-len must cover the longest actual sequence (max page size + 10)")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    for page_size in args.page_sizes:
        for dtype in (torch.float16, torch.bfloat16):
            check(page_size, dtype, args.graph, device, args.heads, args.kv_heads, args.head_dim, args.max_kv_len)
    print("All paged block numerical checks passed.", flush=True)


if __name__ == "__main__":
    main()
