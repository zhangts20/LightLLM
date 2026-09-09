import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices, prepare_chunk_offsets


_CHUNK_SIZE = 64
_MAX_CUBE_BLOCKS = 24


@triton.jit
def _prepare_chunk_matrices(
    Q,
    K,
    G,
    B,
    A,
    ATT,
    GC,
    CU,
    CI,
    H: tl.constexpr,
    HG: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr
):
    ci = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.load(CI + 2 * ci).to(tl.int32)
    ch = tl.load(CI + 2 * ci + 1).to(tl.int32)
    begin = tl.load(CU + seq)
    end = tl.load(CU + seq + 1)
    length = end - begin
    r = tl.arange(0, BT)
    tok = begin + ch * BT + r
    q = tl.load(
        tl.make_block_ptr(
            Q + (begin * HG + head // (H // HG)) * D, (length, D), (HG * D, 1), (ch * BT, 0), (BT, D), (1, 0)
        ),
        boundary_check=(0, 1),
        padding_option="zero",
    )
    k = tl.load(
        tl.make_block_ptr(
            K + (begin * HG + head // (H // HG)) * D, (length, D), (HG * D, 1), (ch * BT, 0), (BT, D), (1, 0)
        ),
        boundary_check=(0, 1),
        padding_option="zero",
    )
    g = tl.load(G + tok * H + head, tok < end, other=0)
    beta = tl.load(B + tok * H + head, tok < end, other=0)
    gc = tl.cumsum(g)
    # A contiguous chunk/head gate vector avoids the padded matrix layout
    # induced by a token-major stride-12 gate store in this fused kernel.
    tl.store(GC + (ci * H + head) * BT + r, gc)
    decay = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0))
    a = tl.dot(k, tl.trans(k), input_precision="ieee") * decay * beta[:, None]
    a = tl.where(r[:, None] > r[None, :], a, 0)
    tl.store(A + (tok[:, None] * H + head) * BT + r[None, :], a, tok[:, None] < end)
    att = tl.dot(q, tl.trans(k), input_precision="ieee") * decay
    att = tl.where(r[:, None] >= r[None, :], att, 0)
    tl.store(ATT + (tok[:, None] * H + head) * BT + r[None, :], att, tok[:, None] < end)


@triton.jit
def _solve_triangular(
    A,
    AI,
    CU,
    CI,
    H: tl.constexpr,
    BT: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    seq = tl.load(CI + 2 * chunk).to(tl.int32)
    local_chunk = tl.load(CI + 2 * chunk + 1).to(tl.int32)
    begin = tl.load(CU + seq) + local_chunk * BT
    end = tl.load(CU + seq + 1)
    r = tl.arange(0, BT)
    ptr = A + ((begin + r[:, None]) * H + head) * BT + r[None, :]
    matrix = tl.load(ptr, begin + r[:, None] < end, other=0)
    matrix = -tl.where(r[:, None] > r[None, :], matrix, 0)
    for row in range(2, tl.minimum(BT, end - begin)):
        # This row has not been updated yet: reuse its original coefficients
        # in UB instead of issuing another GM load on every iteration.
        values = tl.reshape(tl.extract_slice(matrix, [row, 0], [1, BT], [1, 1]), (BT,))
        values += tl.sum(values[:, None] * matrix, axis=0)
        # Rows start on 256-byte boundaries. Updating the row directly avoids
        # the full-matrix boolean broadcasts/casts/selects seen in HIVM IR.
        matrix = tl.insert_slice(matrix, values[None, :], [row, 0], [1, BT], [1, 1])
    matrix += (r[:, None] == r[None, :]).to(tl.float32)
    tl.store(AI + ((begin + r[:, None]) * H + head) * BT + r[None, :], matrix, begin + r[:, None] < end)


@triton.jit
def _recompute_chunk_wu(
    K, V, B, G, A, U, W, CU, CI, VS: tl.constexpr, H: tl.constexpr, HG: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr
):
    ch = tl.program_id(0)
    h = tl.program_id(1)
    seq = tl.load(CI + 2 * ch).to(tl.int32)
    local = tl.load(CI + 2 * ch + 1).to(tl.int32)
    begin = tl.load(CU + seq)
    end = tl.load(CU + seq + 1)
    length = end - begin
    t = begin + local * 64 + tl.arange(0, 64)
    r = tl.arange(0, 64)
    dk = tl.arange(0, DK)
    dv = tl.arange(0, DV)
    a = tl.load(
        tl.make_block_ptr(A + (begin * H + h) * 64, (length, 64), (H * 64, 1), (local * 64, 0), (64, 64), (1, 0)),
        boundary_check=(0, 1),
        padding_option="zero",
    )
    # Cast after loading: do not allocate/copy the full V tensor in FP32.
    v = tl.load(
        tl.make_block_ptr(V + begin * VS + h * DV, (length, DV), (VS, 1), (local * 64, 0), (64, DV), (1, 0)),
        boundary_check=(0, 1),
        padding_option="zero",
    ).to(tl.float32)
    b = tl.load(B + t * H + h, t < end, other=0)
    u = tl.dot(a, v * b[:, None], input_precision="ieee")
    tl.store(U + (t[:, None] * H + h) * DV + dv[None, :], u, t[:, None] < end)
    k = tl.load(
        tl.make_block_ptr(
            K + (begin * HG + h // (H // HG)) * DK, (length, DK), (HG * DK, 1), (local * 64, 0), (64, DK), (1, 0)
        ),
        boundary_check=(0, 1),
        padding_option="zero",
    )
    g = tl.load(G + (ch * H + h) * 64 + r)
    w = tl.dot(a, k * b[:, None] * tl.exp(g)[:, None], input_precision="ieee")
    tl.store(W + (t[:, None] * H + h) * DK + dk[None, :], w, t[:, None] < end)


@triton.jit
def _fused_chunk_recurrence(
    Q,
    KP,
    UP,
    WP,
    GP,
    AP,
    H0,
    OUT,
    CU,
    IDX,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    SEQUENCE_HEADS: tl.constexpr,
    CO,
):
    tile = tl.program_id(0)
    # Explicit scheduling avoids repeated block 0 updates in a partial wave
    # of the installed compiler's automatic block mapping.
    for sh in range(tl.program_id(1), SEQUENCE_HEADS, tl.num_programs(1)):
        seq = sh // H
        head = sh % H
        co = tl.load(CO + seq)
        begin = tl.load(CU + seq)
        end = tl.load(CU + seq + 1)
        si = tl.load(IDX + seq)
        kk = tl.arange(0, K)
        vv = tile * BV + tl.arange(0, BV)
        tt = tl.arange(0, BT)
        state = tl.load(H0 + si * STATE_STRIDE + head * K * V + kk[:, None] * V + vv[None, :])
        for ci in range(tl.cdiv(end - begin, BT)):
            tok = begin + ci * BT + tt
            valid = tok < end
            w = tl.load(WP + (tok[:, None] * H + head) * K + kk[None, :], valid[:, None], other=0)
            u = tl.load(UP + (tok[:, None] * H + head) * V + vv[None, :], valid[:, None], other=0)
            new = u - tl.dot(w, state, input_precision="ieee")
            q = tl.load(Q + (tok[:, None] * HG + head // (H // HG)) * K + kk[None, :], valid[:, None], other=0)
            a = tl.load(AP + (tok[:, None] * H + head) * BT + tt[None, :], valid[:, None], other=0)
            gates = tl.load(GP + ((co + ci) * H + head) * BT + tt)
            out = tl.dot(q, state, input_precision="ieee") * tl.exp(gates)[:, None] + tl.dot(
                a, new, input_precision="ieee"
            )
            tl.store(OUT + (tok[:, None] * H + head) * V + vv[None, :], out * SCALE, valid[:, None])
            glast = tl.load(GP + ((co + ci) * H + head) * BT + tl.minimum(BT, end - begin - ci * BT) - 1)
            new = new * tl.exp(glast - gates)[:, None]
            # Contiguous GM load followed by local transpose avoids the padded
            # strided-load/vdeinterleave path that faulted on Ascend 910B2C.
            k = tl.trans(
                tl.load(KP + (tok[:, None] * HG + head // (H // HG)) * K + kk[None, :], valid[:, None], other=0)
            )
            state = state * tl.exp(glast) + tl.dot(k, new, input_precision="ieee")
        tl.store(H0 + si * STATE_STRIDE + head * K * V + kk[:, None] * V + vv[None, :], state)


def chunk_prefill_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
):
    key_heads, key_dim = k.shape[2:]
    value_heads, value_dim = v.shape[2:]
    q = q.float().contiguous()
    k = k.float().contiguous()
    beta = beta.float().contiguous()
    chunk_indices = prepare_chunk_indices(cu_seqlens, _CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, _CHUNK_SIZE)
    num_chunks = len(chunk_indices)
    grid = (num_chunks, value_heads)
    log_gates = torch.empty(num_chunks, value_heads, _CHUNK_SIZE, device=q.device, dtype=torch.float32)
    lower = torch.empty(1, q.shape[1], value_heads, _CHUNK_SIZE, device=q.device, dtype=torch.float32)
    attention = torch.empty_like(lower)
    inverse = torch.empty_like(lower)
    _prepare_chunk_matrices[grid](
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
        multibuffer=False,
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
    out = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    sequence_heads = (len(cu_seqlens) - 1) * value_heads
    _fused_chunk_recurrence[(1, min(sequence_heads, _MAX_CUBE_BLOCKS))](
        q,
        k,
        values,
        weights,
        log_gates,
        attention,
        initial_state,
        out,
        cu_seqlens,
        state_indices,
        value_heads,
        key_heads,
        key_dim,
        value_dim,
        _CHUNK_SIZE,
        value_dim,
        scale,
        initial_state.stride(0),
        sequence_heads,
        chunk_offsets,
        multibuffer=False,
        unit_flag=True,
    )
    return out
