"""Device checks for exact-token INT8 block decode and the original page path."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[7]

@pytest.fixture(scope="module")
def kernel():
    if not torch.cuda.is_available() or not Path("/opt/maca").exists():
        pytest.skip("MetaX device required")
    path = ROOT / "lightllm/common/basemodel/triton_kernel/att/decode_att/int8kv/maca_int8kv_flash_decoding.py"
    spec = importlib.util.spec_from_file_location("maca_block_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.int8kv_flash_decode


def reference(q, k, ks, v, vs, mapping, lengths, causal):
    k = (k.float().reshape(*k.shape[:2], -1, 8) * ks.float()[..., None]).reshape(k.shape).to(q.dtype)
    v = (v.float().reshape(*v.shape[:2], -1, 8) * vs.float()[..., None]).reshape(v.shape).to(q.dtype)
    outputs = []
    for b, length in enumerate(lengths.tolist()):
        ids = mapping[b, :length].long()
        key = k[ids].repeat_interleave(q.shape[2] // k.shape[1], dim=1)
        value = v[ids].repeat_interleave(q.shape[2] // v.shape[1], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[b].float(), key.float()) / q.shape[-1]**.5
        if causal:
            visible = torch.arange(length, device=q.device)[None, :] <= (length-q.shape[1]+torch.arange(q.shape[1], device=q.device))[:, None]
            scores.masked_fill_(~visible[None], -float("inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), value.float()).to(q.dtype))
    return torch.stack(outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mapped", [False, True])
def test_block_decode_eager_graph_and_mapping_replay(kernel, dtype, mapped):
    torch.manual_seed(37)
    device = "cuda"
    page = 128
    q = torch.randn((2, 5 if mapped else 6, 6, 256), device=device, dtype=dtype)
    k = torch.randint(-100,101,(1024,1,256),device=device,dtype=torch.int8)
    v = torch.randint(-100,101,k.shape,device=device,dtype=torch.int8)
    ks = (torch.rand(1024,1,32,device=device)*.02+.005).to(dtype)
    vs = (torch.rand_like(ks)*.02+.005).to(dtype)
    lengths = torch.tensor([129,273],dtype=torch.int32,device=device)
    req_ids = torch.tensor([2,0],device=device,dtype=torch.int32)
    pages = torch.tensor([[5,1,6],[3,0,7]],device=device,dtype=torch.int32)
    mapping = (pages.long()[...,None]*page+torch.arange(page,device=device)).reshape(2,-1)
    if mapped:
        # Completely non-contiguous slots, including both the prefix and draft
        # rows. A page division would silently produce a different attention.
        mapping = torch.stack([torch.randperm(1024,device=device)[:384] for _ in range(2)])
        table = torch.zeros((3,384),device=device,dtype=torch.int32)
        table[req_ids.long()] = mapping.int()
    else:
        table = pages
    def run():
        return kernel(q,k,ks,v,vs,lengths,page_table=table,page_size=page,
                      token_req_indices=req_ids if mapped else None,causal=not mapped,
                      quant_group_size=8,max_kv_len=532288)
    def check(out):
        expected = reference(q,k,ks,v,vs,mapping,lengths,not mapped)
        torch.testing.assert_close(out,expected,atol=.008 if dtype==torch.bfloat16 else .0015,rtol=.03)
        assert torch.isfinite(out).all()
    check(run())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):out=run()
    graph.replay();torch.cuda.synchronize();check(out)
    if mapped:
        mapping = mapping.flip(1).contiguous()
        table[req_ids.long()] = mapping.int()
    else:
        table.copy_(table.flip(1).contiguous())
        mapping=(table.long()[...,None]*page+torch.arange(page,device=device)).reshape(2,-1)
    lengths.copy_(torch.tensor([257,141],device=device,dtype=torch.int32))
    graph.replay();torch.cuda.synchronize();check(out)


def test_long_noncausal_decode_crosses_all_split_partitions(kernel):
    torch.manual_seed(91)
    dtype, device = torch.bfloat16, "cuda"
    n, length = 131200, 131071
    q = torch.randn((1,5,6,256),device=device,dtype=dtype)
    k = torch.randint(-100,101,(n,1,256),device=device,dtype=torch.int8)
    v = torch.randint(-100,101,k.shape,device=device,dtype=torch.int8)
    ks = (torch.rand(n,1,32,device=device)*.02+.005).to(dtype)
    vs = (torch.rand_like(ks)*.02+.005).to(dtype)
    mapping = torch.randperm(n,device=device).int()[None]
    lengths = torch.tensor([length],device=device,dtype=torch.int32)
    output = kernel(q,k,ks,v,vs,lengths,page_table=mapping,page_size=128,
                    token_req_indices=torch.zeros(1,device=device,dtype=torch.int32),
                    causal=False,max_kv_len=532288)
    expected = reference(q,k,ks,v,vs,mapping,lengths,False)
    torch.testing.assert_close(output,expected,atol=.001,rtol=.03)
    assert torch.isfinite(output).all()
