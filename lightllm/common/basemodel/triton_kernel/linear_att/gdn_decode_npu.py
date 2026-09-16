from __future__ import annotations

import torch
import triton
import triton.language as tl


def _vector_core_count() -> int:
    try:
        return int(triton.runtime.driver.active.utils.get_aivector_core_num())
    except Exception:
        return 40


# ACL-graph replay reuses these; do not torch.empty / .to() every decode step.
_DECODE_OUT_CACHE: dict[tuple, torch.Tensor] = {}


def _cached_empty(device, shape, dtype, name: str) -> torch.Tensor:
    key = (name, device, tuple(shape), dtype)
    buf = _DECODE_OUT_CACHE.get(key)
    if buf is None:
        buf = torch.empty(shape, device=device, dtype=dtype)
        _DECODE_OUT_CACHE[key] = buf
    return buf


def _cached_like(tensor: torch.Tensor, name: str) -> torch.Tensor:
    return _cached_empty(tensor.device, tensor.shape, tensor.dtype, name)


@triton.jit
def _conv_pack_gdn_decode_npu_kernel(
    mixed_qkv,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    q_out,
    k_out,
    v_out,
    stride_m_b: tl.constexpr,
    stride_m_d: tl.constexpr,
    stride_s_b: tl.constexpr,
    stride_s_d: tl.constexpr,
    stride_s_w: tl.constexpr,
    stride_w_d: tl.constexpr,
    stride_w_w: tl.constexpr,
    q_dim: tl.constexpr,
    k_dim: tl.constexpr,
    v_dim: tl.constexpr,
    conv_dim: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES: tl.constexpr,
    TILES_PER_BATCH: tl.constexpr,
):
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        row = tile // TILES_PER_BATCH
        block = tile % TILES_PER_BATCH
        offs = block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < conv_dim
        state_idx = tl.load(conv_state_indices + row).to(tl.int32)

        x = tl.load(mixed_qkv + row * stride_m_b + offs * stride_m_d, mask=mask, other=0.0).to(tl.float32)
        y = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for i in tl.static_range(0, KERNEL_SIZE - 1):
            s = tl.load(
                conv_state + state_idx * stride_s_b + offs * stride_s_d + i * stride_s_w,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            w = tl.load(conv_weight + offs * stride_w_d + i * stride_w_w, mask=mask, other=0.0).to(tl.float32)
            y += s * w
        w = tl.load(
            conv_weight + offs * stride_w_d + (KERNEL_SIZE - 1) * stride_w_w, mask=mask, other=0.0
        ).to(tl.float32)
        y += x * w
        if HAS_BIAS:
            y += tl.load(conv_bias + offs, mask=mask, other=0.0).to(tl.float32)
        if APPLY_SILU:
            y = y * tl.sigmoid(y)

        for i in tl.static_range(0, KERNEL_SIZE - 2):
            next_s = tl.load(
                conv_state + state_idx * stride_s_b + offs * stride_s_d + (i + 1) * stride_s_w,
                mask=mask,
                other=0.0,
            )
            tl.store(
                conv_state + state_idx * stride_s_b + offs * stride_s_d + i * stride_s_w,
                next_s,
                mask=mask,
            )
        tl.store(
            conv_state + state_idx * stride_s_b + offs * stride_s_d + (KERNEL_SIZE - 2) * stride_s_w,
            x,
            mask=mask,
        )

        q_mask = mask & (offs < q_dim)
        k_mask = mask & (offs >= q_dim) & (offs < q_dim + k_dim)
        v_mask = mask & (offs >= q_dim + k_dim)
        tl.store(q_out + row * q_dim + offs, y, mask=q_mask)
        tl.store(k_out + row * k_dim + (offs - q_dim), y, mask=k_mask)
        tl.store(v_out + row * v_dim + (offs - q_dim - k_dim), y, mask=v_mask)


def conv_pack_gdn_decode_inputs_npu(
    mixed_qkv: torch.Tensor,
    z_raw: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state_indices: torch.Tensor,
    activation: str,
    conv_size: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    block_n: int = 256,
):
    batch = mixed_qkv.shape[0]
    q_dim = num_k_heads * head_k_dim
    k_dim = q_dim
    v_dim = num_v_heads * head_v_dim
    conv_dim = q_dim + k_dim + v_dim

    q = _cached_empty(mixed_qkv.device, (batch, 1, num_k_heads, head_k_dim), mixed_qkv.dtype, "pack_q")
    k = _cached_empty(mixed_qkv.device, (batch, 1, num_k_heads, head_k_dim), mixed_qkv.dtype, "pack_k")
    v = _cached_empty(mixed_qkv.device, (batch, 1, num_v_heads, head_v_dim), mixed_qkv.dtype, "pack_v")

    tiles_per_batch = triton.cdiv(conv_dim, block_n)
    tiles = batch * tiles_per_batch
    grid = (min(_vector_core_count(), tiles),)
    _conv_pack_gdn_decode_npu_kernel[grid](
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        conv_state_indices,
        q,
        k,
        v,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        q_dim,
        k_dim,
        v_dim,
        conv_dim,
        conv_size,
        HAS_BIAS=conv_bias is not None,
        APPLY_SILU=activation in ("silu", "swish"),
        BLOCK_N=block_n,
        TILES=tiles,
        TILES_PER_BATCH=tiles_per_batch,
        multibuffer=False,
    )
    z = z_raw.view(batch, num_v_heads, head_v_dim)
    a = a_raw.view(batch, num_v_heads)
    b = b_raw.view(batch, num_v_heads)
    return q, k, v, z, a, b


@triton.jit
def _conv_pack_gdn_decode_npu_contig_kernel(
    mixed_qkv,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    y_out,
    stride_m_b: tl.constexpr,
    stride_y_b: tl.constexpr,
    stride_s_b: tl.constexpr,
    stride_s_w: tl.constexpr,
    stride_w_w: tl.constexpr,
    conv_dim: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES: tl.constexpr,
    TILES_PER_BATCH: tl.constexpr,
):
    # Persistent SGL-style layout: state [slot, 3, dim], weight [4, dim].
    # Channel is the contiguous tail axis so each load is a vector copy.
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        row = tile // TILES_PER_BATCH
        block = tile % TILES_PER_BATCH
        offs = block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < conv_dim
        state_idx = tl.load(conv_state_indices + row).to(tl.int32)

        x = tl.load(mixed_qkv + row * stride_m_b + offs, mask=mask, other=0.0).to(tl.float32)
        s0 = tl.load(conv_state + state_idx * stride_s_b + offs, mask=mask, other=0.0).to(tl.float32)
        s1 = tl.load(conv_state + state_idx * stride_s_b + stride_s_w + offs, mask=mask, other=0.0).to(tl.float32)
        s2 = tl.load(conv_state + state_idx * stride_s_b + 2 * stride_s_w + offs, mask=mask, other=0.0).to(tl.float32)
        w0 = tl.load(conv_weight + offs, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(conv_weight + stride_w_w + offs, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(conv_weight + 2 * stride_w_w + offs, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(conv_weight + 3 * stride_w_w + offs, mask=mask, other=0.0).to(tl.float32)
        y = s0 * w0 + s1 * w1 + s2 * w2 + x * w3
        if HAS_BIAS:
            y += tl.load(conv_bias + offs, mask=mask, other=0.0).to(tl.float32)
        if APPLY_SILU:
            y = y * tl.sigmoid(y)

        tl.store(y_out + row * stride_y_b + offs, y, mask=mask)
        tl.store(conv_state + state_idx * stride_s_b + offs, s1, mask=mask)
        tl.store(conv_state + state_idx * stride_s_b + stride_s_w + offs, s2, mask=mask)
        tl.store(conv_state + state_idx * stride_s_b + 2 * stride_s_w + offs, x, mask=mask)


def _split_packed_y(y: torch.Tensor, num_k_heads: int, head_k_dim: int, num_v_heads: int, head_v_dim: int):
    batch = y.shape[0]
    q_dim = num_k_heads * head_k_dim
    k_dim = q_dim
    q = y[:, :q_dim].view(batch, 1, num_k_heads, head_k_dim)
    k = y[:, q_dim : q_dim + k_dim].view(batch, 1, num_k_heads, head_k_dim)
    v = y[:, q_dim + k_dim :].view(batch, 1, num_v_heads, head_v_dim)
    return q, k, v


def conv_pack_gdn_decode_inputs_npu_contig(
    mixed_qkv: torch.Tensor,
    z_raw: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    conv_state_c: torch.Tensor,
    conv_weight_t: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state_indices: torch.Tensor,
    activation: str,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    block_n: int = 512,
):
    assert conv_state_c.stride(-1) == 1, conv_state_c.stride()
    assert conv_weight_t.stride(-1) == 1, conv_weight_t.stride()
    batch, conv_dim = mixed_qkv.shape
    y = _cached_empty(mixed_qkv.device, (batch, conv_dim), mixed_qkv.dtype, "pack_y")
    tiles_per_batch = triton.cdiv(conv_dim, block_n)
    tiles = batch * tiles_per_batch
    grid = (min(_vector_core_count(), tiles),)
    _conv_pack_gdn_decode_npu_contig_kernel[grid](
        mixed_qkv,
        conv_state_c,
        conv_weight_t,
        conv_bias,
        conv_state_indices,
        y,
        mixed_qkv.stride(0),
        y.stride(0),
        conv_state_c.stride(0),
        conv_state_c.stride(1),
        conv_weight_t.stride(0),
        conv_dim,
        HAS_BIAS=conv_bias is not None,
        APPLY_SILU=activation in ("silu", "swish"),
        BLOCK_N=block_n,
        TILES=tiles,
        TILES_PER_BATCH=tiles_per_batch,
        multibuffer=False,
    )
    q, k, v = _split_packed_y(y, num_k_heads, head_k_dim, num_v_heads, head_v_dim)
    z = z_raw.view(batch, num_v_heads, head_v_dim)
    a = a_raw.view(batch, num_v_heads)
    b = b_raw.view(batch, num_v_heads)
    return q, k, v, z, a, b


@triton.jit
def _fused_gdn_decode_npu_kernel(
    mixed_qkv,
    conv_state,
    conv_weight,
    conv_bias,
    conv_state_indices,
    a_raw,
    b_raw,
    A_log,
    dt_bias,
    ssm_state,
    ssm_state_indices,
    output,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_DIM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    GVA: tl.constexpr,
    stride_m_b: tl.constexpr,
    stride_m_d: tl.constexpr,
    stride_s_b: tl.constexpr,
    stride_s_d: tl.constexpr,
    stride_s_w: tl.constexpr,
    stride_w_d: tl.constexpr,
    stride_w_w: tl.constexpr,
    stride_a_b: tl.constexpr,
    stride_b_b: tl.constexpr,
    stride_ssm: tl.constexpr,
    stride_o_b: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    BLOCKS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    for block in range(tl.program_id(0), BLOCKS, tl.num_programs(0)):
        value_tile = block % NV
        head_batch = block // NV
        row = head_batch // HV
        i_hv = head_batch % HV
        i_h = i_hv // GVA

        conv_idx = tl.load(conv_state_indices + row).to(tl.int32)
        ssm_idx = tl.load(ssm_state_indices + row).to(tl.int32)

        key_offs = tl.arange(0, BK)
        value_offs = value_tile * BV + tl.arange(0, BV)
        key_mask = key_offs < K
        value_mask = value_offs < V

        q_ch = i_h * K + key_offs
        k_ch = Q_DIM + i_h * K + key_offs
        v_ch = Q_DIM + K_DIM + i_hv * V + value_offs
        write_qk = (i_hv % GVA == 0) & (value_tile == 0)

        x_q = tl.load(mixed_qkv + row * stride_m_b + q_ch * stride_m_d, mask=key_mask, other=0.0).to(tl.float32)
        qs0 = tl.load(conv_state + conv_idx * stride_s_b + q_ch * stride_s_d, mask=key_mask, other=0.0).to(tl.float32)
        qs1 = tl.load(
            conv_state + conv_idx * stride_s_b + q_ch * stride_s_d + stride_s_w, mask=key_mask, other=0.0
        ).to(tl.float32)
        qs2 = tl.load(
            conv_state + conv_idx * stride_s_b + q_ch * stride_s_d + 2 * stride_s_w, mask=key_mask, other=0.0
        ).to(tl.float32)
        query = (
            qs0 * tl.load(conv_weight + q_ch * stride_w_d, mask=key_mask, other=0.0).to(tl.float32)
            + qs1 * tl.load(conv_weight + q_ch * stride_w_d + stride_w_w, mask=key_mask, other=0.0).to(tl.float32)
            + qs2 * tl.load(conv_weight + q_ch * stride_w_d + 2 * stride_w_w, mask=key_mask, other=0.0).to(tl.float32)
            + x_q * tl.load(conv_weight + q_ch * stride_w_d + 3 * stride_w_w, mask=key_mask, other=0.0).to(tl.float32)
        )
        if HAS_BIAS:
            query += tl.load(conv_bias + q_ch, mask=key_mask, other=0.0).to(tl.float32)
        if APPLY_SILU:
            query = query * tl.sigmoid(query)
        if write_qk:
            tl.store(conv_state + conv_idx * stride_s_b + q_ch * stride_s_d, qs1, mask=key_mask)
            tl.store(conv_state + conv_idx * stride_s_b + q_ch * stride_s_d + stride_s_w, qs2, mask=key_mask)
            tl.store(conv_state + conv_idx * stride_s_b + q_ch * stride_s_d + 2 * stride_s_w, x_q, mask=key_mask)

        x_k = tl.load(mixed_qkv + row * stride_m_b + k_ch * stride_m_d, mask=key_mask, other=0.0).to(tl.float32)
        ks0 = tl.load(conv_state + conv_idx * stride_s_b + k_ch * stride_s_d, mask=key_mask, other=0.0).to(tl.float32)
        ks1 = tl.load(
            conv_state + conv_idx * stride_s_b + k_ch * stride_s_d + stride_s_w, mask=key_mask, other=0.0
        ).to(tl.float32)
        ks2 = tl.load(
            conv_state + conv_idx * stride_s_b + k_ch * stride_s_d + 2 * stride_s_w, mask=key_mask, other=0.0
        ).to(tl.float32)
        key = (
            ks0 * tl.load(conv_weight + k_ch * stride_w_d, mask=key_mask, other=0.0).to(tl.float32)
            + ks1 * tl.load(conv_weight + k_ch * stride_w_d + stride_w_w, mask=key_mask, other=0.0).to(tl.float32)
            + ks2 * tl.load(conv_weight + k_ch * stride_w_d + 2 * stride_w_w, mask=key_mask, other=0.0).to(tl.float32)
            + x_k * tl.load(conv_weight + k_ch * stride_w_d + 3 * stride_w_w, mask=key_mask, other=0.0).to(tl.float32)
        )
        if HAS_BIAS:
            key += tl.load(conv_bias + k_ch, mask=key_mask, other=0.0).to(tl.float32)
        if APPLY_SILU:
            key = key * tl.sigmoid(key)
        if write_qk:
            tl.store(conv_state + conv_idx * stride_s_b + k_ch * stride_s_d, ks1, mask=key_mask)
            tl.store(conv_state + conv_idx * stride_s_b + k_ch * stride_s_d + stride_s_w, ks2, mask=key_mask)
            tl.store(conv_state + conv_idx * stride_s_b + k_ch * stride_s_d + 2 * stride_s_w, x_k, mask=key_mask)

        x_v = tl.load(mixed_qkv + row * stride_m_b + v_ch * stride_m_d, mask=value_mask, other=0.0).to(tl.float32)
        vs0 = tl.load(conv_state + conv_idx * stride_s_b + v_ch * stride_s_d, mask=value_mask, other=0.0).to(tl.float32)
        vs1 = tl.load(
            conv_state + conv_idx * stride_s_b + v_ch * stride_s_d + stride_s_w, mask=value_mask, other=0.0
        ).to(tl.float32)
        vs2 = tl.load(
            conv_state + conv_idx * stride_s_b + v_ch * stride_s_d + 2 * stride_s_w, mask=value_mask, other=0.0
        ).to(tl.float32)
        value = (
            vs0 * tl.load(conv_weight + v_ch * stride_w_d, mask=value_mask, other=0.0).to(tl.float32)
            + vs1 * tl.load(conv_weight + v_ch * stride_w_d + stride_w_w, mask=value_mask, other=0.0).to(tl.float32)
            + vs2 * tl.load(conv_weight + v_ch * stride_w_d + 2 * stride_w_w, mask=value_mask, other=0.0).to(tl.float32)
            + x_v * tl.load(conv_weight + v_ch * stride_w_d + 3 * stride_w_w, mask=value_mask, other=0.0).to(tl.float32)
        )
        if HAS_BIAS:
            value += tl.load(conv_bias + v_ch, mask=value_mask, other=0.0).to(tl.float32)
        if APPLY_SILU:
            value = value * tl.sigmoid(value)
        tl.store(conv_state + conv_idx * stride_s_b + v_ch * stride_s_d, vs1, mask=value_mask)
        tl.store(conv_state + conv_idx * stride_s_b + v_ch * stride_s_d + stride_s_w, vs2, mask=value_mask)
        tl.store(conv_state + conv_idx * stride_s_b + v_ch * stride_s_d + 2 * stride_s_w, x_v, mask=value_mask)

        query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        query = query * scale

        a = tl.load(a_raw + row * stride_a_b + i_hv).to(tl.float32)
        b = tl.load(b_raw + row * stride_b_b + i_hv).to(tl.float32)
        al = tl.load(A_log + i_hv).to(tl.float32)
        bias = tl.load(dt_bias + i_hv).to(tl.float32)
        z = a + bias
        softplus = tl.where(z <= SOFTPLUS_THRESHOLD, tl.log(1 + tl.exp(z)), z)
        decay = -tl.exp(al) * softplus
        beta = tl.sigmoid(b)

        state_mask = key_mask[:, None] & value_mask[None, :]
        state_ptr = (
            ssm_state + ssm_idx * stride_ssm + i_hv * K * V + key_offs[:, None] * V + value_offs[None, :]
        )
        state = tl.load(state_ptr, mask=state_mask, other=0.0).to(tl.float32)
        state *= tl.exp(decay)
        value -= tl.sum(state * key[:, None], axis=0)
        value *= beta
        state += key[:, None] * value[None, :]
        result = tl.sum(state * query[:, None], axis=0)
        tl.store(state_ptr, state.to(state_ptr.dtype.element_ty), mask=state_mask)
        tl.store(
            output + row * stride_o_b + i_hv * V + value_offs,
            result.to(output.dtype.element_ty),
            mask=value_mask,
        )


def fused_gdn_decode_npu(
    mixed_qkv: torch.Tensor,
    z_raw: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    conv_state_indices: torch.Tensor,
    ssm_states: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    activation: str = "silu",
    block_v: int = 8,
):
    assert conv_weight.shape[1] == 4, "fused NPU decode currently supports conv width=4"
    batch = mixed_qkv.shape[0]
    q_dim = num_k_heads * head_k_dim
    k_dim = q_dim
    gva = num_v_heads // num_k_heads
    scale = head_k_dim**-0.5

    output = torch.empty((batch, 1, num_v_heads, head_v_dim), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    block_k = triton.next_power_of_2(head_k_dim)
    block_v = min(triton.next_power_of_2(head_v_dim), block_v)
    nv = triton.cdiv(head_v_dim, block_v)
    blocks = batch * num_v_heads * nv
    grid = (min(_vector_core_count(), blocks),)

    _fused_gdn_decode_npu_kernel[grid](
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        conv_state_indices,
        a_raw,
        b_raw,
        A_log,
        dt_bias,
        ssm_states,
        ssm_state_indices,
        output,
        scale,
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k_dim,
        V=head_v_dim,
        Q_DIM=q_dim,
        K_DIM=k_dim,
        BK=block_k,
        BV=block_v,
        NV=nv,
        GVA=gva,
        stride_m_b=mixed_qkv.stride(0),
        stride_m_d=mixed_qkv.stride(1),
        stride_s_b=conv_state.stride(0),
        stride_s_d=conv_state.stride(1),
        stride_s_w=conv_state.stride(2),
        stride_w_d=conv_weight.stride(0),
        stride_w_w=conv_weight.stride(1),
        stride_a_b=a_raw.stride(0),
        stride_b_b=b_raw.stride(0),
        stride_ssm=ssm_states.stride(0),
        stride_o_b=output.stride(0),
        HAS_BIAS=conv_bias is not None,
        APPLY_SILU=activation in ("silu", "swish"),
        BLOCKS=blocks,
        SOFTPLUS_THRESHOLD=20.0,
        multibuffer=False,
    )
    z = z_raw.view(batch, num_v_heads, head_v_dim)
    return output, z


@triton.jit
def _fused_recurrent_decode_npu_kernel(
    q,
    k,
    v,
    o,
    h0,
    A_log,
    dt_bias,
    a_raw,
    b_raw,
    ssm_state_indices,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    GVA: tl.constexpr,
    stride_q_tok: tl.constexpr,
    stride_k_tok: tl.constexpr,
    stride_v_tok: tl.constexpr,
    stride_o_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_h0: tl.constexpr,
    stride_idx: tl.constexpr,
    BLOCKS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    # Decode T=1: one logical tile is (batch, value-head, V-tile).
    # Decompose the program-local block id only; do not flatten QKV gathers.
    for block in range(tl.program_id(0), BLOCKS, tl.num_programs(0)):
        value_tile = block % NV
        head_batch = block // NV
        row = head_batch // HV
        i_hv = head_batch % HV
        i_h = i_hv // GVA

        key_offs = tl.arange(0, BK)
        value_offs = value_tile * BV + tl.arange(0, BV)
        key_mask = key_offs < K
        value_mask = value_offs < V
        state_mask = key_mask[:, None] & value_mask[None, :]

        slot = tl.load(ssm_state_indices + row * stride_idx).to(tl.int32)
        query = tl.load(q + row * stride_q_tok + i_h * K + key_offs, mask=key_mask, other=0).to(tl.float32)
        key = tl.load(k + row * stride_k_tok + i_h * K + key_offs, mask=key_mask, other=0).to(tl.float32)
        value = tl.load(v + row * stride_v_tok + i_hv * V + value_offs, mask=value_mask, other=0).to(tl.float32)
        query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        query = query * scale

        a = tl.load(a_raw + row * stride_a_tok + i_hv).to(tl.float32)
        b = tl.load(b_raw + row * stride_b_tok + i_hv).to(tl.float32)
        al = tl.load(A_log + i_hv).to(tl.float32)
        bias = tl.load(dt_bias + i_hv).to(tl.float32)
        z = a + bias
        softplus = tl.where(z <= SOFTPLUS_THRESHOLD, tl.log(1 + tl.exp(z)), z)
        decay = -tl.exp(al) * softplus
        update_scale = tl.sigmoid(b)

        state_ptr = h0 + slot * stride_h0 + i_hv * K * V + key_offs[:, None] * V + value_offs[None, :]
        state = tl.load(state_ptr, mask=state_mask, other=0).to(tl.float32)
        state *= tl.exp(decay)
        value -= tl.sum(state * key[:, None], axis=0)
        value *= update_scale
        state += key[:, None] * value[None, :]
        result = tl.sum(state * query[:, None], axis=0)
        tl.store(state_ptr, state.to(state_ptr.dtype.element_ty), mask=state_mask)
        tl.store(o + row * stride_o_tok + i_hv * V + value_offs, result.to(o.dtype.element_ty), mask=value_mask)


def fused_recurrent_decode_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    out: torch.Tensor | None = None,
    persist: bool = True,
    block_v: int | None = None,
) -> torch.Tensor:
    """910B decode recurrent with gating still inline. T=1 only.

    q/k: ``[B, 1, H, K]``, v/out: ``[B, 1, HV, V]``, a_raw/b_raw: ``[B, HV]``.
    NPU decode serving calls this; CUDA keeps the GPU-grid fused recurrent.

    persist=True pins the grid to Vector-core count and loops leftover tiles
    inside the program. persist=False launches one program per tile.
    """
    assert q.shape[1] == 1 and k.shape[1] == 1 and v.shape[1] == 1
    assert a_raw.stride(-1) == 1 and b_raw.stride(-1) == 1, "a_raw/b_raw must be feat-contiguous"
    batch, _, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    assert value_heads % key_heads == 0
    if out is None:
        out = _cached_like(v, "rec_out")
    block_k = triton.next_power_of_2(key_dim)
    # GPU FLA caps BV at 32. On 910B Vector, a 128x128 fp32 state tile fits
    # UB (~64KB); larger BV cuts persist-loop trips at B>1.
    if block_v is None:
        block_v = min(triton.next_power_of_2(value_dim), 32)
    else:
        block_v = min(triton.next_power_of_2(int(block_v)), triton.next_power_of_2(value_dim))
    value_tiles = triton.cdiv(value_dim, block_v)
    blocks = batch * value_heads * value_tiles
    grid_n = min(_vector_core_count(), blocks) if persist else blocks
    _fused_recurrent_decode_npu_kernel[(grid_n,)](
        q,
        k,
        v,
        out,
        initial_state,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
        ssm_state_indices,
        key_dim**-0.5,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        NV=value_tiles,
        GVA=value_heads // key_heads,
        stride_q_tok=q.stride(0),
        stride_k_tok=k.stride(0),
        stride_v_tok=v.stride(0),
        stride_o_tok=out.stride(0),
        stride_a_tok=a_raw.stride(0),
        stride_b_tok=b_raw.stride(0),
        stride_h0=initial_state.stride(0),
        stride_idx=ssm_state_indices.stride(0),
        BLOCKS=blocks,
        SOFTPLUS_THRESHOLD=20.0,
        multibuffer=False,
    )
    return out
