"""910B4 chunked gated-delta prefill with two state buffers per sequence.

Keep the historical 910B2C kernels in their separate module. Prepare the
triangular chunk solve in parallel, then evaluate residuals, state updates,
and outputs together. FP32 volatile GM reloads separate Cube and Vector uses
that otherwise stalled the installed compiler's mixed pipeline in experiments.
Output arithmetic follows the factored recurrence, so BF16 results need not be
bitwise identical to the affine implementation; full model accuracy is a
separate validation gate.
"""
from typing import List

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .chunk_prefill_npu_910b2c import _CHUNK_SIZE, _prepare_chunk_matrices, _solve_triangular, _recompute_chunk_wu


_VALUE_TILE_SIZE = 128


@triton.jit
def _shared_grams_910b4(
    Q,
    K,
    KK,
    QK,
    CU,
    CI,
    HG: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
):
    ch = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.load(CI + 2 * ch).to(tl.int32)
    local = tl.load(CI + 2 * ch + 1).to(tl.int32)
    begin = tl.load(CU + seq)
    end = tl.load(CU + seq + 1)
    t = begin + local * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    q = tl.load(Q + (t[:, None] * HG + head) * D + d[None, :], t[:, None] < end, other=0)
    k = tl.load(K + (t[:, None] * HG + head) * D + d[None, :], t[:, None] < end, other=0)
    kk = tl.dot(k, tl.trans(k), input_precision="ieee")
    qk = tl.dot(q, tl.trans(k), input_precision="ieee")
    r = tl.arange(0, BT)
    ptr = (ch * HG + head) * BT * BT + r[:, None] * BT + r[None, :]
    tl.store(KK + ptr, kk)
    tl.store(QK + ptr, qk)


@triton.jit
def _apply_chunk_gates_910b4(
    KK,
    QK,
    G,
    B,
    A,
    ATT,
    GC,
    CU,
    CI,
    H: tl.constexpr,
    HG: tl.constexpr,
    BT: tl.constexpr,
):
    ch = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.load(CI + 2 * ch).to(tl.int32)
    local = tl.load(CI + 2 * ch + 1).to(tl.int32)
    begin = tl.load(CU + seq)
    end = tl.load(CU + seq + 1)
    r = tl.arange(0, BT)
    t = begin + local * BT + r
    g = tl.load(G + t * H + head, t < end, other=0)
    b = tl.load(B + t * H + head, t < end, other=0)
    gc = tl.cumsum(g)
    tl.store(GC + (ch * H + head) * BT + r, gc)
    decay = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0))
    ptr = (ch * HG + head // (H // HG)) * BT * BT + r[:, None] * BT + r[None, :]
    kk = tl.load(KK + ptr)
    qk = tl.load(QK + ptr)
    a = tl.where(r[:, None] > r[None, :], kk * decay * b[:, None], 0)
    att = tl.where(r[:, None] >= r[None, :], qk * decay, 0)
    out = (t[:, None] * H + head) * BT + r[None, :]
    tl.store(A + out, a, t[:, None] < end)
    tl.store(ATT + out, att, t[:, None] < end)


def _prepare_chunk_matrices_910b4(
    Q: torch.Tensor,
    K: torch.Tensor,
    G: torch.Tensor,
    B: torch.Tensor,
    A: torch.Tensor,
    ATT: torch.Tensor,
    GC: torch.Tensor,
    CU: torch.Tensor,
    CI: torch.Tensor    ,
    H: int,
    HG: int,
    D: int,
    BT: int,
):
    # Q/K heads are shared by several value heads. Compute each Gram matrix
    # once per Q/K head; apply the distinct gates and beta values afterwards.
    count = CI.shape[0]
    if H == HG:
        _prepare_chunk_matrices[(count, H)](Q, K, G, B, A, ATT, GC, CU, CI, H, HG, D, BT, multibuffer=False)
        return
    kk = torch.empty((count, HG, BT, BT), device=Q.device, dtype=torch.float32)
    qk = torch.empty_like(kk)
    _shared_grams_910b4[(count, HG)](Q, K, kk, qk, CU, CI, HG, D, BT, multibuffer=False)
    _apply_chunk_gates_910b4[(count, H)](kk, qk, G, B, A, ATT, GC, CU, CI, H, HG, BT, multibuffer=False)




@triton.jit
def _copy_initial_ring_states(
    H0,
    STATES,
    IDX,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    SS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    tile = tl.program_id(1)
    x = tile * BLOCK + tl.arange(0, BLOCK)
    si = tl.load(IDX + seq)
    value = tl.load(H0 + si * SS + x, x < H * D * V, other=0)
    tl.store(STATES + (seq * 2) * H * D * V + x, value, x < H * D * V)


@triton.jit
def _copy_final_ring_states(
    H0,
    STATES,
    CO,
    IDX,
    H: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    SS: tl.constexpr,
    BLOCK: tl.constexpr
):
    seq = tl.program_id(0)
    tile = tl.program_id(1)
    x = tile * BLOCK + tl.arange(0, BLOCK)
    si = tl.load(IDX + seq)
    count = tl.load(CO + seq + 1) - tl.load(CO + seq)
    co = seq * 2 + count % 2
    value = tl.load(STATES + co * H * D * V + x, x < H * D * V, other=0)
    tl.store(H0 + si * SS + x, value, x < H * D * V)


@triton.jit
def _chunk_recurrence_output_910b4(
    KP,
    UP,
    WP,
    GP,
    STATES,
    CU,
    CO,
    RP,
    Q,
    AP,
    OUT,
    SCALE:tl.constexpr,
    OSTRIDE:tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    SH: tl.constexpr,
    OFFSET: tl.constexpr,
    BV: tl.constexpr,
):
    block = OFFSET + tl.program_id(0)
    tile = block // SH
    sh = block % SH
    seq = sh // H
    h = sh % H
    begin = tl.load(CU + seq)
    end = tl.load(CU + seq + 1)
    co = tl.load(CO + seq)
    d = tl.arange(0,D)
    v = tile * BV + tl.arange(0,BV)
    t = tl.arange(0,64)
    for ci in range(tl.cdiv(end-begin,64)):
        tok = begin + ci*64+t
        valid = tok < end
        state = tl.load(STATES + ((seq*2+ci%2)*H+h)*D*V+d[:,None]*V+v[None,:])
        w = tl.load(WP+(tok[:,None]*H+h)*D+d[None,:],valid[:,None],other=0)
        u = tl.load(UP+(tok[:,None]*H+h)*V+v[None,:],valid[:,None],other=0)
        residual = u - tl.dot(w,state,input_precision="ieee")
        tl.store(RP+(tok[:,None]*H+h)*V+v[None,:],residual,valid[:,None])
        gates=tl.load(GP+((co+ci)*H+h)*64+t)
        last=tl.minimum(64,end-begin-ci*64)-1
        glast=tl.load(GP+((co+ci)*H+h)*64+last)
        k=tl.load(KP+(tok[:,None]*HG+h//(H//HG))*D+d[None,:],valid[:,None],other=0)
        # Separate Cube input and Vector use through explicit GM reloads.
        state_vector = tl.load(STATES + ((seq*2+ci%2)*H+h)*D*V+d[:,None]*V+v[None,:], volatile=True)
        residual_cube = tl.load(RP+(tok[:,None]*H+h)*V+v[None,:],valid[:,None],other=0,volatile=True)
        updated = state_vector*tl.exp(glast)+tl.dot(tl.trans(k),residual_cube*tl.exp(glast-gates)[:,None],input_precision="ieee")
        tl.store(STATES+((seq*2+(ci+1)%2)*H+h)*D*V+d[:,None]*V+v[None,:],updated)
        qout=tl.load(Q+(tok[:,None]*HG+h//(H//HG))*D+d[None,:],valid[:,None],other=0)
        old_state=tl.load(STATES+((seq*2+ci%2)*H+h)*D*V+d[:,None]*V+v[None,:],volatile=True)
        rout=tl.load(RP+(tok[:,None]*H+h)*V+v[None,:],valid[:,None],other=0,volatile=True)
        att=tl.load(AP+(tok[:,None]*H+h)*64+t[None,:],valid[:,None],other=0)
        result=tl.dot(qout,old_state,input_precision="ieee")*tl.exp(gates)[:,None]+tl.dot(att,rout,input_precision="ieee")
        tl.store(OUT+tok[:,None]*OSTRIDE+h*V+v[None,:],result*SCALE,valid[:,None])




def _all_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
    output=None,
):
    key_heads, key_dim = k.shape[2:]
    value_heads, value_dim = v.shape[2:]
    q = q.float().contiguous()
    k = k.float().contiguous()
    beta = beta.float().contiguous()
    chunk_indices = prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, _CHUNK_SIZE)
    num_chunks = len(chunk_indices)
    if num_chunks == 0:
        return torch.empty_like(v)
    grid = (num_chunks, value_heads)
    log_gates = torch.empty(num_chunks, value_heads, _CHUNK_SIZE, device=q.device, dtype=torch.float32)
    lower = torch.empty(1, q.shape[1], value_heads, _CHUNK_SIZE, device=q.device, dtype=torch.float32)
    attention = torch.empty_like(lower)
    inverse = torch.empty_like(lower)
    _prepare_chunk_matrices_910b4(
        q,
        k,
        g,
        beta,
        lower,
        attention,
        log_gates,
        cu_seqlens,
        chunk_indices,
        value_heads,
        key_heads,
        key_dim,
        _CHUNK_SIZE,
    )
    _solve_triangular[grid](lower, inverse, cu_seqlens, chunk_indices, value_heads, _CHUNK_SIZE, multibuffer=False)
    del lower
    values = torch.empty(v.shape, device=v.device, dtype=torch.float32)
    weights = torch.empty(1, q.shape[1], value_heads, key_dim, device=q.device, dtype=torch.float32)
    _recompute_chunk_wu[grid](
        k,
        v,
        beta,
        log_gates,
        inverse,
        values,
        weights,
        cu_seqlens,
        chunk_indices,
        v.stride(1),
        value_heads,
        key_heads,
        key_dim,
        value_dim,
        multibuffer=False,
    )
    del inverse
    out = output if output is not None else torch.empty(v.shape, device=v.device, dtype=v.dtype)
    states = torch.empty((2*(len(cu_seqlens)-1),value_heads,key_dim,value_dim),device=v.device,dtype=torch.float32)
    _copy_initial_ring_states[(len(cu_seqlens) - 1, triton.cdiv(value_heads * key_dim * value_dim, 1024))](
        initial_state,
        states,
        state_indices,
        value_heads,
        key_dim,
        value_dim,
        initial_state.stride(0),
        1024,
    )
    residuals = values
    sh = (len(cu_seqlens) - 1) * value_heads
    # Use the actual Cube count (20 on 910B4). Do not let automatic block
    # mapping replay a partial wave against mutable recurrent state.
    cores = triton.runtime.driver.active.utils.get_aicore_num()
    value_tile = min(_VALUE_TILE_SIZE, value_dim)
    for offset in range(0, triton.cdiv(value_dim, value_tile) * sh, cores):
        blocks = min(cores, triton.cdiv(value_dim, value_tile) * sh - offset)
        _chunk_recurrence_output_910b4[(blocks,)](k,values,weights,log_gates,states,cu_seqlens,chunk_offsets,residuals,q,attention,out,scale,out.stride(1),value_heads,key_heads,key_dim,value_dim,sh,offset,value_tile,multibuffer=False)
    _copy_final_ring_states[(len(cu_seqlens) - 1, triton.cdiv(value_heads * key_dim * value_dim, 1024))](
        initial_state,
        states,
        chunk_offsets,
        state_indices,
        value_heads,
        key_dim,
        value_dim,
        initial_state.stride(0),
        1024,
    )
    return out

from functools import lru_cache
from .utils import tensor_cache

_WINDOW_TOKENS = 2048

@lru_cache(maxsize=128)
def _window_cu(device: torch.device, length: int, dtype: torch.dtype) -> torch.Tensor:
    # Immutable metadata; its identity also permits the existing chunk-index
    # cache to reuse the small index tensors across layers.
    return torch.tensor([0, length], device=device, dtype=dtype)

@lru_cache(maxsize=128)
def _packed_window_cu(device: torch.device, offsets: List[int], dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(offsets, device=device, dtype=dtype)

@tensor_cache
def _packed_window_plan(cu_seqlens: torch.Tensor) -> tuple[tuple[int, int, int, int, torch.Tensor], ...]:
    offsets = cu_seqlens.detach().cpu().tolist()
    plan = []
    start = 0
    total = offsets[-1]
    while start < total:
        end = min(total, start + _WINDOW_TOKENS)
        # Round only inside the sequence containing the boundary. This uses
        # at most 63 extra tokens and preserves its original 64-token chunks.
        for begin_seq, end_seq in zip(offsets, offsets[1:]):
            if begin_seq < end < end_seq:
                end = min(end_seq, begin_seq + ((end - begin_seq + 63) // 64) * 64)
                break
        active = [i for i,(a,b) in enumerate(zip(offsets,offsets[1:])) if min(b,end)>max(a,start)]
        first,last = active[0],active[-1]+1
        local = [0]
        for i in range(first,last):
            local.append(local[-1] + max(0,min(offsets[i+1],end)-max(offsets[i],start)))
        cu = _packed_window_cu(cu_seqlens.device,tuple(local),cu_seqlens.dtype)
        plan.append((first,last,start,end,cu))
        start = end
    return tuple(plan)

def chunk_prefill_npu_910b4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    n = q.shape[1]
    if n <= _WINDOW_TOKENS or q.shape[-1] != 128 or v.shape[-1] != 128:
        return _all_heads(q,k,v,g,beta,initial_state,cu_seqlens,state_indices,scale)
    if len(cu_seqlens) == 2:
        plan = ((0,1,start,min(n,start+_WINDOW_TOKENS),_window_cu(q.device,min(n-start,_WINDOW_TOKENS),cu_seqlens.dtype)) for start in range(0,n,_WINDOW_TOKENS))
    else:
        plan = _packed_window_plan(cu_seqlens)
    out = torch.empty_like(v)
    for first,last,start,end,cu in plan:
        # Windows begin at multiples of 64 within each sequence, preserving
        # the exact recurrence order and chunk boundaries. The terminal state
        # of each window becomes the initial state of its successor.
        _all_heads(q[:,start:end],k[:,start:end],v[:,start:end],g[:,start:end],beta[:,start:end],initial_state,cu,state_indices[first:last],scale,output=out[:,start:end])
    return out
