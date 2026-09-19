import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K,
    Dest_loc,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_g,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_g,
    stride_o_d,
    stride_os_bs,
    stride_os_h,
    stride_os_g,
    group_size,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)

    src_data = tl.load(
        K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :],
        mask=offs_g[:, None] < group_size,
        other=0.0,
    )
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0).to(Out_scale.dtype.element_ty)
    q_src_data = (src_data / data_scale[:, None]).to(tl.int8)

    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return


@torch.no_grad()
def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale, quant_group_dim):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert triton.next_power_of_2(quant_group_dim) == quant_group_dim, "error quant group dim"

    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    grid = (seq_len, head_num)
    num_warps = 1

    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(Out.shape[0], Out.shape[1], group_size, group_dim)

    _fwd_kernel_destindex_copy_quantize_kv[grid](
        K,
        DestLoc,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out.stride(3),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        group_size,
        BLOCK_GROUP_NUM=triton.next_power_of_2(group_size),
        BLOCK_GROUP_DIM=group_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_dequantize_int8kv(
    k,
    k_ss,
    k_sh,
    k_sg,
    k_sd,
    k_scale,
    k_scale_ss,
    k_scale_sh,
    k_scale_sg,
    k_scale_sd,
    v,
    v_ss,
    v_sh,
    v_sg,
    v_sd,
    v_scale,
    v_scale_ss,
    v_scale_sh,
    v_scale_sg,
    v_scale_sd,
    req_to_token_indexs,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    b_seq_len,
    b_req_idx,
    b_kv_start_loc,
    k_out,
    k_out_ss,
    k_out_sh,
    k_out_sg,
    k_out_sd,
    v_out,
    v_out_ss,
    v_out_sh,
    v_out_sg,
    v_out_sd,
    k_head_num,
    v_head_num,
    group_count,
    group_dim,
    SEQ_BLOCK_SIZE: tl.constexpr,
    GROUP_COUNT_BLOCK_SIZE: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr,
):
    start_block_index = tl.program_id(0)
    cur_batch = tl.program_id(1)
    cur_batch_req_idx = tl.load(b_req_idx + cur_batch)
    cur_seq_len = tl.load(b_seq_len + cur_batch)
    if start_block_index * SEQ_BLOCK_SIZE >= cur_seq_len:
        return

    out_start_loc = tl.load(b_kv_start_loc + cur_batch)

    offs_kv_loc = (start_block_index * SEQ_BLOCK_SIZE + tl.arange(0, SEQ_BLOCK_SIZE)) % cur_seq_len
    kv_loc = tl.load(req_to_token_indexs + cur_batch_req_idx * stride_req_to_tokens_b + offs_kv_loc).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_GROUP_DIM) % group_dim
    offs_scale_d = tl.arange(0, 1)
    group_offs = tl.arange(0, GROUP_COUNT_BLOCK_SIZE) % group_count

    for k_head_index in tl.range(0, k_head_num, step=1, num_stages=3):
        k_int8 = tl.load(
            k
            + kv_loc[:, None, None] * k_ss
            + k_head_index * k_sh
            + group_offs[None, :, None] * k_sg
            + offs_d[None, None, :]
        )
        k_scale_data = tl.load(
            k_scale
            + kv_loc[:, None, None] * k_scale_ss
            + k_head_index * k_scale_sh
            + group_offs[None, :, None] * k_scale_sg
            + offs_scale_d[None, None, :]
        )
        k_out_data = k_int8.to(k_out.dtype.element_ty) * k_scale_data
        tl.store(
            k_out
            + (out_start_loc + offs_kv_loc[:, None, None]) * k_out_ss
            + k_head_index * k_out_sh
            + group_offs[None, :, None] * k_out_sg
            + offs_d[None, None, :],
            k_out_data,
        )

    for v_head_index in tl.range(0, v_head_num, step=1, num_stages=3):
        v_int8 = tl.load(
            v
            + kv_loc[:, None, None] * v_ss
            + v_head_index * v_sh
            + group_offs[None, :, None] * v_sg
            + offs_d[None, None, :]
        )
        v_scale_data = tl.load(
            v_scale
            + kv_loc[:, None, None] * v_scale_ss
            + v_head_index * v_scale_sh
            + group_offs[None, :, None] * v_scale_sg
            + offs_scale_d[None, None, :]
        )
        v_out_data = v_int8.to(v_out.dtype.element_ty) * v_scale_data
        tl.store(
            v_out
            + (out_start_loc + offs_kv_loc[:, None, None]) * v_out_ss
            + v_head_index * v_out_sh
            + group_offs[None, :, None] * v_out_sg
            + offs_d[None, None, :],
            v_out_data,
        )
    return


@torch.no_grad()
def dequantize_int8kv(
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_kv_start_loc: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    max_len_in_batch: int,
    quant_group_size: int,
):
    batch_size = b_seq_len.shape[0]
    k_head_num = k.shape[1]
    k_head_dim = k.shape[2]
    v_head_num = v.shape[1]
    v_head_dim = v.shape[2]
    assert k_head_dim % quant_group_size == 0, "error head dim, can not been supported to copy quant kv"
    assert v_head_dim % quant_group_size == 0, "error head dim, can not been supported to copy quant kv"
    assert k_head_dim == v_head_dim, "error head dim, can not been supported to copy quant kv"
    assert k_head_dim // v_scale.shape[2] == quant_group_size, "error head dim, can not been supported to copy quant kv"
    assert k_head_dim in [64, 128, 256]

    group_count = k_head_dim // quant_group_size
    group_dim = quant_group_size

    k = k.view((k.shape[0], k.shape[1], group_count, group_dim))
    v = v.view((v.shape[0], v.shape[1], group_count, group_dim))
    k_scale = k_scale.view((k_scale.shape[0], k_scale.shape[1], group_count, 1))
    v_scale = v_scale.view((v_scale.shape[0], v_scale.shape[1], group_count, 1))

    # 使拆分的grid 具有足够的并行度
    SEQ_BLOCK_SIZE = 128
    while triton.cdiv(max_len_in_batch, SEQ_BLOCK_SIZE) * batch_size < 512:
        SEQ_BLOCK_SIZE = SEQ_BLOCK_SIZE // 2
        if SEQ_BLOCK_SIZE <= 1:
            break

    if SEQ_BLOCK_SIZE <= 1:
        SEQ_BLOCK_SIZE = 8

    grid = (triton.cdiv(max_len_in_batch, SEQ_BLOCK_SIZE), batch_size)
    num_warps = 4
    k_out = k_out.view((k_out.shape[0], k_out.shape[1], group_count, group_dim))
    v_out = v_out.view((v_out.shape[0], v_out.shape[1], group_count, group_dim))

    _fwd_dequantize_int8kv[grid](
        k=k,
        k_ss=k.stride(0),
        k_sh=k.stride(1),
        k_sg=k.stride(2),
        k_sd=k.stride(3),
        k_scale=k_scale,
        k_scale_ss=k_scale.stride(0),
        k_scale_sh=k_scale.stride(1),
        k_scale_sg=k_scale.stride(2),
        k_scale_sd=k_scale.stride(2),
        v=v,
        v_ss=v.stride(0),
        v_sh=v.stride(1),
        v_sg=v.stride(2),
        v_sd=v.stride(3),
        v_scale=v_scale,
        v_scale_ss=v_scale.stride(0),
        v_scale_sh=v_scale.stride(1),
        v_scale_sg=v_scale.stride(2),
        v_scale_sd=v_scale.stride(3),
        req_to_token_indexs=req_to_token_indexs,
        stride_req_to_tokens_b=req_to_token_indexs.stride(0),
        stride_req_to_tokens_s=req_to_token_indexs.stride(1),
        b_seq_len=b_seq_len,
        b_req_idx=b_req_idx,
        b_kv_start_loc=b_kv_start_loc,
        k_out=k_out,
        k_out_ss=k_out.stride(0),
        k_out_sh=k_out.stride(1),
        k_out_sg=k_out.stride(2),
        k_out_sd=k_out.stride(3),
        v_out=v_out,
        v_out_ss=v_out.stride(0),
        v_out_sh=v_out.stride(1),
        v_out_sg=v_out.stride(2),
        v_out_sd=v_out.stride(3),
        k_head_num=k_head_num,
        v_head_num=v_head_num,
        group_count=group_count,
        group_dim=group_dim,
        SEQ_BLOCK_SIZE=SEQ_BLOCK_SIZE,
        GROUP_COUNT_BLOCK_SIZE=triton.next_power_of_2(group_count),
        BLOCK_GROUP_DIM=triton.next_power_of_2(group_dim),
        num_warps=num_warps,
        num_stages=1,
    )
    return


def _align_block_d(head_dim: int, group_size: int) -> int:
    block_d = 1 if head_dim <= 1 else 1 << (head_dim - 1).bit_length()
    while block_d % group_size != 0:
        block_d *= 2
    return block_d


@triton.jit
def _gather_dequant_int8kv_npu(
    K,
    K_scale,
    V,
    V_scale,
    Indices,
    K_out,
    V_out,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_ks_t,
    stride_ks_h,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_vs_t,
    stride_vs_h,
    stride_ko_t,
    stride_ko_h,
    stride_ko_d,
    stride_vo_t,
    stride_vo_h,
    stride_vo_d,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Per-token scale, D-tiled. Smaller UB than loading full head_dim.
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = offs_t < n_tokens
    d_mask = offs_d < HEAD_DIM
    locs = tl.load(Indices + offs_t, mask=t_mask, other=0).to(tl.int64)
    k_s = tl.load(K_scale + locs * stride_ks_t + head * stride_ks_h, mask=t_mask, other=0.0)
    k_i8 = tl.load(
        K + locs[:, None] * stride_k_t + head * stride_k_h + offs_d[None, :] * stride_k_d,
        mask=t_mask[:, None] & d_mask[None, :],
        other=0,
    )
    k = k_i8.to(K_out.dtype.element_ty) * k_s[:, None].to(K_out.dtype.element_ty)
    tl.store(
        K_out + offs_t[:, None] * stride_ko_t + head * stride_ko_h + offs_d[None, :] * stride_ko_d,
        k,
        mask=t_mask[:, None] & d_mask[None, :],
    )
    v_s = tl.load(V_scale + locs * stride_vs_t + head * stride_vs_h, mask=t_mask, other=0.0)
    v_i8 = tl.load(
        V + locs[:, None] * stride_v_t + head * stride_v_h + offs_d[None, :] * stride_v_d,
        mask=t_mask[:, None] & d_mask[None, :],
        other=0,
    )
    v = v_i8.to(V_out.dtype.element_ty) * v_s[:, None].to(V_out.dtype.element_ty)
    tl.store(
        V_out + offs_t[:, None] * stride_vo_t + head * stride_vo_h + offs_d[None, :] * stride_vo_d,
        v,
        mask=t_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _gather_dequant_int8kv(
    K,
    K_scale,
    V,
    V_scale,
    Indices,
    K_out,
    V_out,
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
    stride_ko_t,
    stride_ko_h,
    stride_ko_d,
    stride_vo_t,
    stride_vo_h,
    stride_vo_d,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    offs_t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1)
    t_mask = offs_t < n_tokens
    locs = tl.load(Indices + offs_t, mask=t_mask, other=0).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D)
    offs_g = tl.arange(0, NUM_GROUPS)
    d_mask = offs_d < HEAD_DIM
    g_mask = offs_g < (HEAD_DIM // GROUP_SIZE)

    k_i8 = tl.load(
        K + locs[:, None] * stride_k_t + head * stride_k_h + offs_d[None, :] * stride_k_d,
        mask=t_mask[:, None] & d_mask[None, :],
        other=0,
    )
    k_s = tl.load(
        K_scale + locs[:, None] * stride_ks_t + head * stride_ks_h + offs_g[None, :] * stride_ks_g,
        mask=t_mask[:, None] & g_mask[None, :],
        other=0.0,
    )
    k_i8 = tl.reshape(k_i8, (BLOCK_T, NUM_GROUPS, GROUP_SIZE))
    k_s = tl.reshape(k_s, (BLOCK_T, NUM_GROUPS, 1))
    k = tl.reshape(k_i8.to(K_out.dtype.element_ty) * k_s.to(K_out.dtype.element_ty), (BLOCK_T, BLOCK_D))
    tl.store(
        K_out + offs_t[:, None] * stride_ko_t + head * stride_ko_h + offs_d[None, :] * stride_ko_d,
        k,
        mask=t_mask[:, None] & d_mask[None, :],
    )

    v_i8 = tl.load(
        V + locs[:, None] * stride_v_t + head * stride_v_h + offs_d[None, :] * stride_v_d,
        mask=t_mask[:, None] & d_mask[None, :],
        other=0,
    )
    v_s = tl.load(
        V_scale + locs[:, None] * stride_vs_t + head * stride_vs_h + offs_g[None, :] * stride_vs_g,
        mask=t_mask[:, None] & g_mask[None, :],
        other=0.0,
    )
    v_i8 = tl.reshape(v_i8, (BLOCK_T, NUM_GROUPS, GROUP_SIZE))
    v_s = tl.reshape(v_s, (BLOCK_T, NUM_GROUPS, 1))
    v = tl.reshape(v_i8.to(V_out.dtype.element_ty) * v_s.to(V_out.dtype.element_ty), (BLOCK_T, BLOCK_D))
    tl.store(
        V_out + offs_t[:, None] * stride_vo_t + head * stride_vo_h + offs_d[None, :] * stride_vo_d,
        v,
        mask=t_mask[:, None] & d_mask[None, :],
    )


@torch.no_grad()
def gather_dequant_int8kv(
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    indices: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    quant_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_tokens = int(indices.numel())
    if n_tokens == 0:
        return k_out, v_out
    n_kv = k.shape[1]
    head_dim = k.shape[2]
    if head_dim % quant_group_size != 0:
        raise ValueError(f"head_dim {head_dim} is not divisible by group {quant_group_size}")
    on_npu = k.device.type == "npu"
    if on_npu and quant_group_size == head_dim:
        # 910B: T=32/D=256, multibuffer off. T=64 corrupts; T=128 UB-align faults.
        block_t = 32 if n_tokens >= 32 else (16 if n_tokens >= 16 else 8)
        block_d = 256
        grid = (triton.cdiv(n_tokens, block_t), n_kv, triton.cdiv(head_dim, block_d))
        _gather_dequant_int8kv_npu[grid](
            k,
            k_scale,
            v,
            v_scale,
            indices,
            k_out,
            v_out,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k_scale.stride(0),
            k_scale.stride(1) if k_scale.ndim > 1 else 0,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v_scale.stride(0),
            v_scale.stride(1) if v_scale.ndim > 1 else 0,
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            v_out.stride(0),
            v_out.stride(1),
            v_out.stride(2),
            n_tokens,
            HEAD_DIM=head_dim,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            multibuffer=False,
        )
        return k_out, v_out
    block_d = _align_block_d(head_dim, quant_group_size)
    num_groups = block_d // quant_group_size
    block_t = 64 if n_tokens >= 64 else 16
    grid = (triton.cdiv(n_tokens, block_t), n_kv)
    _gather_dequant_int8kv[grid](
        k,
        k_scale,
        v,
        v_scale,
        indices,
        k_out,
        v_out,
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
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        n_tokens,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        GROUP_SIZE=quant_group_size,
        NUM_GROUPS=num_groups,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=2,
    )
    return k_out, v_out


def test2():
    import time

    B, N_CTX, H, D = 1, 3, 12, 128
    src = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda()
    dest_loc = torch.arange(0, B * N_CTX, dtype=torch.int32).cuda()
    value_dest = torch.randn((B * N_CTX, H, D), dtype=torch.float16).cuda().to(torch.int8)
    scale_dest = torch.randn((B * N_CTX, H, D // 8), dtype=torch.float16).cuda()

    for _ in range(10):
        destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_quantize_kv(src, dest_loc, value_dest, scale_dest)
    torch.cuda.synchronize()
    t2 = time.time()

    print("Time cost ", t2 - t1)
    value_dest = value_dest.view((B * N_CTX, H, D // 8, 8))
    scale_dest = scale_dest.view((B * N_CTX, H, D // 8, 1))
    print("max ", torch.max(torch.abs((value_dest * scale_dest).view(B * N_CTX, H, D) - src)))
    print("mean ", torch.mean(torch.abs((value_dest * scale_dest).view(B * N_CTX, H, D) - src)))
    cos = torch.nn.CosineSimilarity(0)
    print("cos ", cos(src.flatten().to(torch.float32), (value_dest * scale_dest).flatten().to(torch.float32)))
