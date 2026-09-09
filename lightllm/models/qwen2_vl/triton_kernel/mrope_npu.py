import torch
import triton
import triton.language as tl


def can_use_mrope_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_rotary_factor: float,
) -> bool:
    rotary_dim = int(q.shape[2] * partial_rotary_factor)
    return (
        q.dtype in (torch.bfloat16, torch.float16)
        and rotary_dim in (32, 64, 128, 256)
        and q.shape[0] >= 1024
    )


@triton.jit
def _mrope_prefill_kernel(
    TOKENS: tl.constexpr,
    BT: tl.constexpr,
    q,
    k,
    Cos,
    Sin,
    mrope_section,
    stride_cosld,
    stride_cosd,
    stride_sinld,
    stride_sind,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    is_interleaved: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    for task in range(tl.program_id(0), (HEAD_Q + HEAD_K) * tl.cdiv(TOKENS, BT), 24):
        head_index = task // tl.cdiv(TOKENS, BT)
        seq_index = ((task % tl.cdiv(TOKENS, BT)) * BT + tl.arange(0, BT))[:, None]
        valid = seq_index < TOKENS

        dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)[None, :]
        dim_range1 = dim_range0 + BLOCK_DMODEL // 2

        t_cos = Cos + seq_index * stride_cosd
        h_cos = Cos + stride_cosld + seq_index * stride_cosd
        w_cos = Cos + 2 * stride_cosld + seq_index * stride_cosd
        t_sin = Sin + seq_index * stride_sind
        h_sin = Sin + stride_sinld + seq_index * stride_sind
        w_sin = Sin + 2 * stride_sinld + seq_index * stride_sind

        mrope_section_t = tl.load(mrope_section + 0)
        mrope_section_h = tl.load(mrope_section + 1)
        mrope_section_w = tl.load(mrope_section + 2)

        # Updated offsets for half head_dim
        offsets = tl.arange(0, BLOCK_DMODEL // 2)[None, :]
        if is_interleaved:
            h_mask = ((offsets % 3) == 1) & (offsets <= 3 * mrope_section_h)
            w_mask = ((offsets % 3) == 2) & (offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
        else:
            t_end = mrope_section_t
            h_end = t_end + mrope_section_h
            t_mask = offsets < mrope_section_t
            h_mask = (t_end <= offsets) & (offsets < h_end)
            w_mask = (h_end <= offsets) & (offsets < BLOCK_DMODEL // 2)

        t_cos = tl.load(t_cos + offsets, mask=t_mask & valid, other=0)
        t_sin = tl.load(t_sin + offsets, mask=t_mask & valid, other=0)
        h_cos = tl.load(h_cos + offsets, mask=h_mask & valid, other=0)
        h_sin = tl.load(h_sin + offsets, mask=h_mask & valid, other=0)
        w_cos = tl.load(w_cos + offsets, mask=w_mask & valid, other=0)
        w_sin = tl.load(w_sin + offsets, mask=w_mask & valid, other=0)

        cos = t_cos + h_cos + w_cos
        sin = t_sin + h_sin + w_sin

        if head_index < HEAD_Q:
            q_head_index = head_index
            off_q0 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range0 * stride_qd
            off_q1 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range1 * stride_qd
            q0 = tl.load(q + off_q0, mask=valid, other=0)
            q1 = tl.load(q + off_q1, mask=valid, other=0)
            out_q0 = q0 * cos - q1 * sin
            out_q1 = q0 * sin + q1 * cos
            tl.store(q + off_q0, out_q0, mask=valid)
            tl.store(q + off_q1, out_q1, mask=valid)
        else:
            k_head_index = head_index - HEAD_Q
            off_k0 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range0 * stride_kd
            off_k1 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range1 * stride_kd

            k0 = tl.load(k + off_k0, mask=valid, other=0)
            k1 = tl.load(k + off_k1, mask=valid, other=0)

            out_k0 = k0 * cos - k1 * sin
            out_k1 = k0 * sin + k1 * cos

            tl.store(k + off_k0, out_k0, mask=valid)
            tl.store(k + off_k1, out_k1, mask=valid)

    return


@torch.no_grad()
def mrope_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: int,
    is_interleaved: bool = True,
    partial_rotary_factor: float = 0.25,
) -> None:
    rotary_dim = int(q.shape[2] * partial_rotary_factor)
    # Limit the tile area: 256 rotary dimensions at BT64 exceed local UB.
    _mrope_prefill_kernel[(24,)](
        TOKENS=q.shape[0],
        BT=min(64, 8192 // rotary_dim),
        q=q,
        k=k,
        Cos=cos,
        Sin=sin,
        mrope_section=mrope_section,
        stride_cosld=cos.stride(0),
        stride_cosd=cos.stride(1),
        stride_sinld=sin.stride(0),
        stride_sind=sin.stride(1),
        stride_qbs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        is_interleaved=is_interleaved,
        HEAD_Q=q.shape[1],
        HEAD_K=k.shape[1],
        BLOCK_DMODEL=rotary_dim,
    )


@triton.jit
def _mrope_npu_small_kernel(
    q,
    k,
    Cos,
    Sin,
    mrope_section,
    stride_cosld,
    stride_cosd,
    stride_sinld,
    stride_sind,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    is_interleaved: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    TOKENS: tl.constexpr,
):
    # Keep logical programs within the physical grid: auto-blockify's tail can
    # otherwise wrap to head zero/token zero and apply the in-place rotation twice.
    for task in range(tl.program_id(0), (HEAD_Q + HEAD_K) * TOKENS, tl.num_programs(0)):
        head_index = task // TOKENS
        seq_index = task % TOKENS

        dim_range0 = tl.arange(0, BLOCK_DMODEL // 2)
        dim_range1 = dim_range0 + BLOCK_DMODEL // 2

        t_cos = Cos + seq_index * stride_cosd
        h_cos = Cos + stride_cosld + seq_index * stride_cosd
        w_cos = Cos + 2 * stride_cosld + seq_index * stride_cosd
        t_sin = Sin + seq_index * stride_sind
        h_sin = Sin + stride_sinld + seq_index * stride_sind
        w_sin = Sin + 2 * stride_sinld + seq_index * stride_sind

        mrope_section_t = tl.load(mrope_section + 0)
        mrope_section_h = tl.load(mrope_section + 1)
        mrope_section_w = tl.load(mrope_section + 2)

        # Updated offsets for half head_dim
        offsets = tl.arange(0, BLOCK_DMODEL // 2)
        if is_interleaved:
            h_mask = ((offsets % 3) == 1) & (offsets <= 3 * mrope_section_h)
            w_mask = ((offsets % 3) == 2) & (offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
        else:
            t_end = mrope_section_t
            h_end = t_end + mrope_section_h
            t_mask = offsets < mrope_section_t
            h_mask = (t_end <= offsets) & (offsets < h_end)
            w_mask = (h_end <= offsets) & (offsets < BLOCK_DMODEL // 2)

        t_cos = tl.load(t_cos + offsets, mask=t_mask, other=0)
        t_sin = tl.load(t_sin + offsets, mask=t_mask, other=0)
        h_cos = tl.load(h_cos + offsets, mask=h_mask, other=0)
        h_sin = tl.load(h_sin + offsets, mask=h_mask, other=0)
        w_cos = tl.load(w_cos + offsets, mask=w_mask, other=0)
        w_sin = tl.load(w_sin + offsets, mask=w_mask, other=0)

        cos = t_cos + h_cos + w_cos
        sin = t_sin + h_sin + w_sin

        if head_index < HEAD_Q:
            q_head_index = head_index
            off_q0 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range0 * stride_qd
            off_q1 = seq_index * stride_qbs + q_head_index * stride_qh + dim_range1 * stride_qd
            q0 = tl.load(q + off_q0)
            q1 = tl.load(q + off_q1)
            out_q0 = q0 * cos - q1 * sin
            out_q1 = q0 * sin + q1 * cos
            tl.store(q + off_q0, out_q0)
            tl.store(q + off_q1, out_q1)
        else:
            k_head_index = head_index - HEAD_Q
            off_k0 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range0 * stride_kd
            off_k1 = seq_index * stride_kbs + k_head_index * stride_kh + dim_range1 * stride_kd

            k0 = tl.load(k + off_k0)
            k1 = tl.load(k + off_k1)

            out_k0 = k0 * cos - k1 * sin
            out_k1 = k0 * sin + k1 * cos

            tl.store(k + off_k0, out_k0)
            tl.store(k + off_k1, out_k1)

    return


@torch.no_grad()
def mrope_small(q, k, cos, sin, mrope_section, is_interleaved, partial_rotary_factor):
    from triton.runtime import driver

    heads_q, heads_k = q.shape[1], k.shape[1]
    rotary = int(q.shape[2] * partial_rotary_factor)
    tokens = q.shape[0]
    if tokens == 0:
        return
    cores = driver.active.utils.get_aivector_core_num()
    _mrope_npu_small_kernel[(min((heads_q + heads_k) * tokens, cores),)](
        q,
        k,
        cos,
        sin,
        mrope_section,
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        is_interleaved,
        heads_q,
        heads_k,
        rotary,
        tokens,
    )
