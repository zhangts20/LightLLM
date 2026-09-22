import torch
import triton
import triton.language as tl


@triton.jit
def _copy_linear_att_state_to_kv_buffer(
    gpu_conv_ptr,  # uint8 view: [linear_layer_num, req_num, conv_dim, gpu_conv_row_bytes]
    gpu_ssm_ptr,  # uint8 view: [linear_layer_num, req_num * (mtp_step + 1), ssm_bytes]
    cpu_kv_conv_ptr,  # uint8 view: [buffer_num, linear_layer_num, conv_dim * cpu_conv_row_bytes]
    cpu_kv_ssm_ptr,  # uint8 view: [buffer_num, linear_layer_num, ssm_bytes]
    b_req_idx,  # [batch_size,]
    big_page_buffer_ids,  # [batch_size,]
    gpu_conv_stride_l,
    gpu_conv_stride_s,
    gpu_conv_stride_c,
    gpu_conv_stride_d,
    gpu_ssm_stride_l,
    gpu_ssm_stride_s,
    gpu_ssm_stride_d,
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_l,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_l,
    cpu_kv_ssm_stride_d,
    mtp_step,
    gpu_conv_dim,  # number of conv rows
    gpu_conv_tail_dim_bytes,  # bytes copied per conv row; equals the CPU/cache row width
    gpu_ssm_tail_dim,
    BLOCK: tl.constexpr,
):
    cur_layer = tl.program_id(0).to(tl.int64)
    cur_batch = tl.program_id(1).to(tl.int64)
    gpu_conv_stride_l = tl.cast(gpu_conv_stride_l, dtype=tl.int64)
    gpu_conv_stride_s = tl.cast(gpu_conv_stride_s, dtype=tl.int64)
    gpu_conv_stride_c = tl.cast(gpu_conv_stride_c, dtype=tl.int64)
    gpu_conv_stride_d = tl.cast(gpu_conv_stride_d, dtype=tl.int64)
    gpu_ssm_stride_l = tl.cast(gpu_ssm_stride_l, dtype=tl.int64)
    gpu_ssm_stride_s = tl.cast(gpu_ssm_stride_s, dtype=tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, dtype=tl.int64)
    cpu_kv_conv_stride_l = tl.cast(cpu_kv_conv_stride_l, dtype=tl.int64)
    cpu_kv_conv_stride_d = tl.cast(cpu_kv_conv_stride_d, dtype=tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, dtype=tl.int64)
    cpu_kv_ssm_stride_l = tl.cast(cpu_kv_ssm_stride_l, dtype=tl.int64)
    gpu_conv_tail_dim_bytes = tl.cast(gpu_conv_tail_dim_bytes, dtype=tl.int64)

    big_page_buffer_idx = tl.load(big_page_buffer_ids + cur_batch)
    if big_page_buffer_idx == -1:
        return

    cur_req_idx = tl.load(b_req_idx + cur_batch).to(tl.int64)
    cur_state_req_idx = (cur_req_idx * (mtp_step + 1)).to(tl.int64)

    gpu_conv_base = gpu_conv_ptr + cur_layer * gpu_conv_stride_l + cur_req_idx * gpu_conv_stride_s
    cpu_conv_base = cpu_kv_conv_ptr + big_page_buffer_idx * cpu_kv_conv_stride_s + cur_layer * cpu_kv_conv_stride_l
    conv_tail_dim = gpu_conv_dim * gpu_conv_tail_dim_bytes
    for i in range(tl.cdiv(conv_tail_dim, BLOCK)):
        conv_start = i * BLOCK + tl.arange(0, BLOCK)
        conv_row = conv_start // gpu_conv_tail_dim_bytes
        conv_col = conv_start % gpu_conv_tail_dim_bytes
        mask = conv_start < conv_tail_dim
        conv_data = tl.load(gpu_conv_base + conv_row * gpu_conv_stride_c + conv_col, mask=mask)
        tl.store(cpu_conv_base + conv_start, conv_data, mask=mask)

    for i in range(tl.cdiv(gpu_ssm_tail_dim, BLOCK)):
        gpu_start_off = i * BLOCK + tl.arange(0, BLOCK)
        mask = gpu_start_off < gpu_ssm_tail_dim
        ssm_data = tl.load(
            gpu_ssm_ptr + cur_layer * gpu_ssm_stride_l + cur_state_req_idx * gpu_ssm_stride_s + gpu_start_off,
            mask=mask,
        )
        dest_ssm_ptr = (
            cpu_kv_ssm_ptr + big_page_buffer_idx * cpu_kv_ssm_stride_s + cur_layer * cpu_kv_ssm_stride_l + gpu_start_off
        )
        tl.store(dest_ssm_ptr, ssm_data, mask=mask)

    return


_NPU_CORES = None
_NPU_STAGING = {}


def _npu_vector_core_count() -> int:
    global _NPU_CORES
    if _NPU_CORES is None:
        try:
            _NPU_CORES = int(triton.runtime.driver.active.utils.get_aivector_core_num())
        except Exception:
            _NPU_CORES = 40
    return _NPU_CORES


def _npu_staging(batch, layer_num, conv_u32, ssm_u32, device):
    key = (batch, layer_num, conv_u32, ssm_u32)
    hit = _NPU_STAGING.get(key)
    if hit is None:
        hit = (
            torch.empty((batch, layer_num, conv_u32), dtype=torch.uint32, device=device),
            torch.empty((batch, layer_num, ssm_u32), dtype=torch.uint32, device=device),
        )
        _NPU_STAGING[key] = hit
    return hit


def _as_u32_keep2(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(dtype=torch.uint8).view(tensor.shape[0], tensor.shape[1], -1).view(dtype=torch.uint32)


@triton.jit
def _copy_linear_att_state_to_kv_buffer_npu(
    gpu_conv_ptr,
    gpu_ssm_ptr,
    dst_conv_ptr,
    dst_ssm_ptr,
    b_req_idx,
    gpu_conv_stride_l,
    gpu_conv_stride_s,
    gpu_ssm_stride_l,
    gpu_ssm_stride_s,
    dst_conv_stride_b,
    dst_conv_stride_l,
    dst_ssm_stride_b,
    dst_ssm_stride_l,
    mtp_step,
    batch,
    conv_u32,
    ssm_u32,
    conv_tiles,
    tiles_per_lb,
    tiles,
    BLOCK: tl.constexpr,
):
    gpu_conv_stride_l = tl.cast(gpu_conv_stride_l, dtype=tl.int64)
    gpu_conv_stride_s = tl.cast(gpu_conv_stride_s, dtype=tl.int64)
    gpu_ssm_stride_l = tl.cast(gpu_ssm_stride_l, dtype=tl.int64)
    gpu_ssm_stride_s = tl.cast(gpu_ssm_stride_s, dtype=tl.int64)
    dst_conv_stride_b = tl.cast(dst_conv_stride_b, dtype=tl.int64)
    dst_conv_stride_l = tl.cast(dst_conv_stride_l, dtype=tl.int64)
    dst_ssm_stride_b = tl.cast(dst_ssm_stride_b, dtype=tl.int64)
    dst_ssm_stride_l = tl.cast(dst_ssm_stride_l, dtype=tl.int64)
    conv_u32 = tl.cast(conv_u32, dtype=tl.int64)
    ssm_u32 = tl.cast(ssm_u32, dtype=tl.int64)

    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        lb = tile // tiles_per_lb
        sub = tile % tiles_per_lb
        cur_layer = (lb // batch).to(tl.int64)
        cur_batch = (lb % batch).to(tl.int64)
        cur_req_idx = tl.load(b_req_idx + cur_batch).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        if sub < conv_tiles:
            col = sub * BLOCK + offs
            mask = col < conv_u32
            src = gpu_conv_ptr + cur_layer * gpu_conv_stride_l + cur_req_idx * gpu_conv_stride_s
            dst = dst_conv_ptr + cur_batch * dst_conv_stride_b + cur_layer * dst_conv_stride_l
            tl.store(dst + col, tl.load(src + col, mask=mask), mask=mask)
        else:
            col = (sub - conv_tiles) * BLOCK + offs
            mask = col < ssm_u32
            ssm_req = (cur_req_idx * (mtp_step + 1)).to(tl.int64)
            src = gpu_ssm_ptr + cur_layer * gpu_ssm_stride_l + ssm_req * gpu_ssm_stride_s
            dst = dst_ssm_ptr + cur_batch * dst_ssm_stride_b + cur_layer * dst_ssm_stride_l
            tl.store(dst + col, tl.load(src + col, mask=mask), mask=mask)


def _launch_copy_linear_att_state_to_kv_buffer_npu(
    b_req_idx: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    gpu_conv_state: torch.Tensor,
    gpu_ssm_state: torch.Tensor,
    cpu_kv_conv_state: torch.Tensor,
    cpu_kv_ssm_state: torch.Tensor,
    mtp_step: int,
):
    buf_ids = big_page_buffer_ids.detach().cpu().tolist()
    valid_pos = [i for i, buf_id in enumerate(buf_ids) if buf_id >= 0]
    slots = [buf_ids[i] for i in valid_pos]
    batch = len(slots)
    if batch != b_req_idx.numel():
        pick = torch.tensor(valid_pos, device=b_req_idx.device, dtype=torch.int64)
        b_req_idx = b_req_idx.index_select(0, pick).contiguous()

    conv_u32 = cpu_kv_conv_state[0, 0].numel() * cpu_kv_conv_state.element_size() // 4
    ssm_u32 = gpu_ssm_state[0, 0].numel() * gpu_ssm_state.element_size() // 4
    layer_num = gpu_conv_state.shape[0]
    gpu_conv_u32 = _as_u32_keep2(gpu_conv_state)
    gpu_ssm_u32 = _as_u32_keep2(gpu_ssm_state)

    BLOCK = 1024
    conv_tiles = triton.cdiv(conv_u32, BLOCK)
    tiles_per_lb = conv_tiles + triton.cdiv(ssm_u32, BLOCK)
    tiles = layer_num * batch * tiles_per_lb
    dst_conv, dst_ssm = _npu_staging(batch, layer_num, conv_u32, ssm_u32, gpu_conv_state.device)
    _copy_linear_att_state_to_kv_buffer_npu[(min(_npu_vector_core_count(), tiles),)](
        gpu_conv_ptr=gpu_conv_u32,
        gpu_ssm_ptr=gpu_ssm_u32,
        dst_conv_ptr=dst_conv,
        dst_ssm_ptr=dst_ssm,
        b_req_idx=b_req_idx,
        gpu_conv_stride_l=gpu_conv_u32.stride(0),
        gpu_conv_stride_s=gpu_conv_u32.stride(1),
        gpu_ssm_stride_l=gpu_ssm_u32.stride(0),
        gpu_ssm_stride_s=gpu_ssm_u32.stride(1),
        dst_conv_stride_b=dst_conv.stride(0),
        dst_conv_stride_l=dst_conv.stride(1),
        dst_ssm_stride_b=dst_ssm.stride(0),
        dst_ssm_stride_l=dst_ssm.stride(1),
        mtp_step=mtp_step,
        batch=batch,
        conv_u32=conv_u32,
        ssm_u32=ssm_u32,
        conv_tiles=conv_tiles,
        tiles_per_lb=tiles_per_lb,
        tiles=tiles,
        BLOCK=BLOCK,
        multibuffer=False,
    )

    dst_conv = dst_conv.view(dtype=cpu_kv_conv_state.dtype).reshape(batch, *cpu_kv_conv_state.shape[1:])
    dst_ssm = dst_ssm.view(dtype=cpu_kv_ssm_state.dtype).reshape(batch, *cpu_kv_ssm_state.shape[1:])
    for i, slot in enumerate(slots):
        cpu_kv_conv_state[slot].copy_(dst_conv[i])
        cpu_kv_ssm_state[slot].copy_(dst_ssm[i])


def copy_linear_att_state_to_kv_buffer(
    b_req_idx: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    gpu_conv_state: torch.Tensor,  # [linear_layer_num, req_num, conv_dim, kernel_size]
    gpu_ssm_state: torch.Tensor,  # [linear_layer_num, req_num * (mtp_step + 1), ...]
    cpu_kv_conv_state: torch.Tensor,  # [buffer_num, linear_layer_num, conv_dim, kernel_size]
    cpu_kv_ssm_state: torch.Tensor,  # [buffer_num, linear_layer_num, ...]
    mtp_step: int,
):
    if gpu_conv_state.device.type == "npu":
        _launch_copy_linear_att_state_to_kv_buffer_npu(
            b_req_idx=b_req_idx,
            big_page_buffer_ids=big_page_buffer_ids,
            gpu_conv_state=gpu_conv_state,
            gpu_ssm_state=gpu_ssm_state,
            cpu_kv_conv_state=cpu_kv_conv_state,
            cpu_kv_ssm_state=cpu_kv_ssm_state,
            mtp_step=mtp_step,
        )
        return

    # gpu_conv_state 的后两维可能是不连续的。
    assert len(b_req_idx) == big_page_buffer_ids.shape[0]
    BLOCK = 4096

    assert gpu_conv_state.dim() == 4, "gpu_conv_state must be [layer, s, conv_dim, widened_width]"
    assert cpu_kv_conv_state.dim() == 4, "cpu_kv_conv_state must be [size, layer, conv_dim, width_narrow]"
    # 因为存在mtp模式，gpu_conv_state 的最后一个维度可能存在冗余的部分，需要进行切片对齐。
    gpu_conv_state = gpu_conv_state[:, :, :, : cpu_kv_conv_state.shape[-1]]
    gpu_conv_state = gpu_conv_state.view(
        gpu_conv_state.shape[0], gpu_conv_state.shape[1], gpu_conv_state.shape[2], -1
    ).view(dtype=torch.uint8)
    cpu_kv_conv_state = cpu_kv_conv_state.view(cpu_kv_conv_state.shape[0], cpu_kv_conv_state.shape[1], -1).view(
        dtype=torch.uint8
    )
    gpu_ssm_state = gpu_ssm_state.view(gpu_ssm_state.shape[0], gpu_ssm_state.shape[1], -1).view(dtype=torch.uint8)
    cpu_kv_ssm_state = cpu_kv_ssm_state.view(cpu_kv_ssm_state.shape[0], cpu_kv_ssm_state.shape[1], -1).view(
        dtype=torch.uint8
    )
    assert gpu_ssm_state.shape[-1] == cpu_kv_ssm_state.shape[-1]

    gpu_conv_dim = gpu_conv_state.shape[2]
    gpu_conv_tail_dim_bytes = gpu_conv_state.shape[3]

    assert gpu_conv_tail_dim_bytes * gpu_conv_dim == cpu_kv_conv_state.shape[-1]

    assert (
        gpu_conv_state.stride(-1)
        == gpu_ssm_state.stride(-1)
        == cpu_kv_conv_state.stride(-1)
        == cpu_kv_ssm_state.stride(-1)
        == 1
    )
    gpu_ssm_tail_dim = gpu_ssm_state.shape[-1]
    layer_num = gpu_conv_state.shape[0]
    grid = (layer_num, b_req_idx.shape[0])

    _copy_linear_att_state_to_kv_buffer[grid](
        gpu_conv_ptr=gpu_conv_state,
        gpu_ssm_ptr=gpu_ssm_state,
        cpu_kv_conv_ptr=cpu_kv_conv_state,
        cpu_kv_ssm_ptr=cpu_kv_ssm_state,
        b_req_idx=b_req_idx,
        big_page_buffer_ids=big_page_buffer_ids,
        gpu_conv_stride_l=gpu_conv_state.stride(0),
        gpu_conv_stride_s=gpu_conv_state.stride(1),
        gpu_conv_stride_c=gpu_conv_state.stride(2),
        gpu_conv_stride_d=gpu_conv_state.stride(3),
        gpu_ssm_stride_l=gpu_ssm_state.stride(0),
        gpu_ssm_stride_s=gpu_ssm_state.stride(1),
        gpu_ssm_stride_d=gpu_ssm_state.stride(2),
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_l=cpu_kv_conv_state.stride(1),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(2),
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_l=cpu_kv_ssm_state.stride(1),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(2),
        mtp_step=mtp_step,
        gpu_conv_dim=gpu_conv_dim,
        gpu_conv_tail_dim_bytes=gpu_conv_tail_dim_bytes,
        gpu_ssm_tail_dim=gpu_ssm_tail_dim,
        BLOCK=BLOCK,
    )
