import torch
import triton
import triton.language as tl
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.common.basemodel.triton_kernel.linear_att_copy import _npu_vector_core_count


_NPU_STAGING = {}


def _npu_page_staging(batch, att_u32, conv_u32, ssm_u32, device):
    key = (batch, att_u32, conv_u32, ssm_u32, str(device))
    hit = _NPU_STAGING.get(key)
    if hit is None:
        hit = tuple(
            torch.empty((batch, n), dtype=torch.uint32, device=device) for n in (att_u32, conv_u32, ssm_u32, conv_u32, ssm_u32)
        )
        _NPU_STAGING[key] = hit
    return hit


def _gpu_kv_as_u32(gpu_kv: torch.Tensor):
    gpu = gpu_kv
    if not gpu.is_contiguous():
        gpu = gpu.permute(1, 0, 2)
    if not gpu.is_contiguous():
        raise AssertionError("gpu kv must share storage with a contiguous [layer, token, dim] buffer")
    gpu_u32 = gpu.view(dtype=torch.uint32).view(gpu.shape[0], gpu.shape[1], -1)
    if gpu is gpu_kv:
        return gpu_u32, gpu_u32.stride(0), gpu_u32.stride(1), gpu_u32.stride(2)
    return gpu_u32, gpu_u32.stride(1), gpu_u32.stride(0), gpu_u32.stride(2)


def _npu_copy_meta(gpu_kv, cpu_kv_conv, cpu_kv_ssm, big_page_token_num):
    gpu_u32, stride_s, stride_l, stride_d = _gpu_kv_as_u32(gpu_kv)
    att_u32 = gpu_u32.shape[-1] * gpu_kv.shape[1] * big_page_token_num
    conv_u32 = cpu_kv_conv[0].numel() * cpu_kv_conv.element_size() // 4
    ssm_u32 = cpu_kv_ssm[0].numel() * cpu_kv_ssm.element_size() // 4
    return gpu_u32, stride_s, stride_l, stride_d, att_u32, conv_u32, ssm_u32


def _pick_valid_pages(page_indexes, page_readies, big_page_buffer_ids, mem_indexes, big_page_token_num):
    page_ids = page_indexes.detach().cpu().tolist()
    big_ids = big_page_buffer_ids.detach().cpu().tolist()
    if page_readies is None:
        readies = [False] * len(page_ids)
    else:
        readies = page_readies.detach().cpu().tolist()
    valid_pos = []
    valid_pages = []
    valid_big = []
    for i, (page_id, ready, big_id) in enumerate(zip(page_ids, readies, big_ids)):
        if page_id < 0 or ready:
            continue
        valid_pos.append(i)
        valid_pages.append(int(page_id))
        valid_big.append(int(big_id))
    if not valid_pos:
        return [], [], None
    idx = torch.tensor(valid_pos, device=mem_indexes.device, dtype=torch.int64)
    token = mem_indexes.view(len(page_ids), big_page_token_num).index_select(0, idx).reshape(-1).contiguous()
    return valid_pages, valid_big, token


@triton.jit
def _copy_kv_to_cpu_cache_npu(
    mem_indexes_ptr,
    gpu_kv,
    gpu_stride_s,
    gpu_stride_l,
    gpu_stride_d,
    dest_att,
    dest_att_stride_p,
    src_conv,
    dest_conv,
    conv_stride_s,
    dest_conv_stride_p,
    src_ssm,
    dest_ssm,
    ssm_stride_s,
    dest_ssm_stride_p,
    att_u32,
    conv_u32,
    ssm_u32,
    att_tiles,
    conv_tiles,
    tiles_per_page,
    tiles,
    big_page_token_num,
    full_att_layer_num,
    BLOCK: tl.constexpr,
):
    gpu_stride_s = tl.cast(gpu_stride_s, tl.int64)
    gpu_stride_l = tl.cast(gpu_stride_l, tl.int64)
    gpu_stride_d = tl.cast(gpu_stride_d, tl.int64)
    dest_att_stride_p = tl.cast(dest_att_stride_p, tl.int64)
    conv_stride_s = tl.cast(conv_stride_s, tl.int64)
    dest_conv_stride_p = tl.cast(dest_conv_stride_p, tl.int64)
    ssm_stride_s = tl.cast(ssm_stride_s, tl.int64)
    dest_ssm_stride_p = tl.cast(dest_ssm_stride_p, tl.int64)
    att_u32 = tl.cast(att_u32, tl.int64)
    conv_u32 = tl.cast(conv_u32, tl.int64)
    ssm_u32 = tl.cast(ssm_u32, tl.int64)
    offs = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        page = (tile // tiles_per_page).to(tl.int64)
        sub = tile % tiles_per_page
        if sub < att_tiles:
            col = sub * BLOCK + offs
            mask = col < att_u32
            safe_col = tl.where(mask, col, 0)
            per_token = att_u32 // big_page_token_num
            per_layer = per_token // full_att_layer_num
            mem_offs = safe_col // per_token
            mem_index = tl.load(mem_indexes_ptr + page * big_page_token_num + mem_offs, mask=mask, other=-1).to(
                tl.int64
            )
            valid = mask & (mem_index >= 0)
            safe_index = tl.where(valid, mem_index, 0)
            layer = (safe_col // per_layer) % full_att_layer_num
            dim = safe_col % per_layer
            data = tl.load(
                gpu_kv + safe_index * gpu_stride_s + layer * gpu_stride_l + dim * gpu_stride_d,
                mask=valid,
                other=0,
            )
            tl.store(dest_att + page * dest_att_stride_p + safe_col, data, mask=valid)
        else:
            if sub < att_tiles + conv_tiles:
                col = (sub - att_tiles) * BLOCK + offs
                mask = col < conv_u32
                safe_col = tl.where(mask, col, 0)
                data = tl.load(src_conv + page * conv_stride_s + safe_col, mask=mask, other=0)
                tl.store(dest_conv + page * dest_conv_stride_p + safe_col, data, mask=mask)
            else:
                col = (sub - att_tiles - conv_tiles) * BLOCK + offs
                mask = col < ssm_u32
                safe_col = tl.where(mask, col, 0)
                data = tl.load(src_ssm + page * ssm_stride_s + safe_col, mask=mask, other=0)
                tl.store(dest_ssm + page * dest_ssm_stride_p + safe_col, data, mask=mask)


@triton.jit
def _copy_cpu_cache_to_kv_npu(
    mem_indexes_ptr,
    gpu_kv,
    gpu_stride_s,
    gpu_stride_l,
    gpu_stride_d,
    src_att,
    src_att_stride_p,
    src_conv,
    dest_conv,
    conv_stride_s,
    dest_conv_stride_p,
    src_ssm,
    dest_ssm,
    ssm_stride_s,
    dest_ssm_stride_p,
    att_u32,
    conv_u32,
    ssm_u32,
    att_tiles,
    conv_tiles,
    tiles_per_page,
    tiles,
    big_page_token_num,
    full_att_layer_num,
    BLOCK: tl.constexpr,
):
    gpu_stride_s = tl.cast(gpu_stride_s, tl.int64)
    gpu_stride_l = tl.cast(gpu_stride_l, tl.int64)
    gpu_stride_d = tl.cast(gpu_stride_d, tl.int64)
    src_att_stride_p = tl.cast(src_att_stride_p, tl.int64)
    conv_stride_s = tl.cast(conv_stride_s, tl.int64)
    dest_conv_stride_p = tl.cast(dest_conv_stride_p, tl.int64)
    ssm_stride_s = tl.cast(ssm_stride_s, tl.int64)
    dest_ssm_stride_p = tl.cast(dest_ssm_stride_p, tl.int64)
    att_u32 = tl.cast(att_u32, tl.int64)
    conv_u32 = tl.cast(conv_u32, tl.int64)
    ssm_u32 = tl.cast(ssm_u32, tl.int64)
    offs = tl.arange(0, BLOCK)
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        page = (tile // tiles_per_page).to(tl.int64)
        sub = tile % tiles_per_page
        if sub < att_tiles:
            col = sub * BLOCK + offs
            mask = col < att_u32
            safe_col = tl.where(mask, col, 0)
            per_token = att_u32 // big_page_token_num
            per_layer = per_token // full_att_layer_num
            mem_offs = safe_col // per_token
            mem_index = tl.load(mem_indexes_ptr + page * big_page_token_num + mem_offs, mask=mask, other=-1).to(
                tl.int64
            )
            valid = mask & (mem_index >= 0)
            safe_index = tl.where(valid, mem_index, 0)
            layer = (safe_col // per_layer) % full_att_layer_num
            dim = safe_col % per_layer
            data = tl.load(src_att + page * src_att_stride_p + safe_col, mask=valid, other=0)
            tl.store(
                gpu_kv + safe_index * gpu_stride_s + layer * gpu_stride_l + dim * gpu_stride_d,
                data,
                mask=valid,
            )
        else:
            if sub < att_tiles + conv_tiles:
                col = (sub - att_tiles) * BLOCK + offs
                mask = col < conv_u32
                safe_col = tl.where(mask, col, 0)
                data = tl.load(src_conv + page * conv_stride_s + safe_col, mask=mask, other=0)
                tl.store(dest_conv + page * dest_conv_stride_p + safe_col, data, mask=mask)
            else:
                col = (sub - att_tiles - conv_tiles) * BLOCK + offs
                mask = col < ssm_u32
                safe_col = tl.where(mask, col, 0)
                data = tl.load(src_ssm + page * ssm_stride_s + safe_col, mask=mask, other=0)
                tl.store(dest_ssm + page * dest_ssm_stride_p + safe_col, data, mask=mask)


def _launch_copy_kv_to_cpu_cache_npu(
    mem_indexes,
    page_indexes,
    page_readies,
    big_page_buffer_ids,
    cpu_cache_full_att,
    cpu_cache_conv,
    cpu_cache_ssm,
    gpu_kv_full_att_state,
    cpu_kv_conv_state,
    cpu_kv_ssm_state,
    tp_rank,
    big_page_token_num,
    head_scale_size,
):
    valid_pages, valid_big, token = _pick_valid_pages(
        page_indexes, page_readies, big_page_buffer_ids, mem_indexes, big_page_token_num
    )
    if not valid_pages:
        return
    write_full_att = 1 if tp_rank % head_scale_size == 0 else 0
    att_head = tp_rank // head_scale_size
    device = gpu_kv_full_att_state.device
    gpu_u32, gpu_stride_s, gpu_stride_l, gpu_stride_d, att_u32, conv_u32, ssm_u32 = _npu_copy_meta(
        gpu_kv_full_att_state, cpu_kv_conv_state, cpu_kv_ssm_state, big_page_token_num
    )
    batch = len(valid_pages)
    dest_att, dest_conv, dest_ssm, src_conv, src_ssm = _npu_page_staging(batch, att_u32, conv_u32, ssm_u32, device)
    for i, big_id in enumerate(valid_big):
        src_conv[i].copy_(cpu_kv_conv_state[big_id].view(dtype=torch.uint32).reshape(-1))
        src_ssm[i].copy_(cpu_kv_ssm_state[big_id].view(dtype=torch.uint32).reshape(-1))
    BLOCK = 1024
    att_tiles = triton.cdiv(att_u32, BLOCK) if write_full_att else 0
    conv_tiles = triton.cdiv(conv_u32, BLOCK)
    tiles_per_page = att_tiles + conv_tiles + triton.cdiv(ssm_u32, BLOCK)
    tiles = batch * tiles_per_page
    _copy_kv_to_cpu_cache_npu[(min(_npu_vector_core_count(), tiles),)](
        mem_indexes_ptr=token,
        gpu_kv=gpu_u32,
        gpu_stride_s=gpu_stride_s,
        gpu_stride_l=gpu_stride_l,
        gpu_stride_d=gpu_stride_d,
        dest_att=dest_att,
        dest_att_stride_p=dest_att.stride(0),
        src_conv=src_conv,
        dest_conv=dest_conv,
        conv_stride_s=src_conv.stride(0),
        dest_conv_stride_p=dest_conv.stride(0),
        src_ssm=src_ssm,
        dest_ssm=dest_ssm,
        ssm_stride_s=src_ssm.stride(0),
        dest_ssm_stride_p=dest_ssm.stride(0),
        att_u32=att_u32,
        conv_u32=conv_u32,
        ssm_u32=ssm_u32,
        att_tiles=att_tiles,
        conv_tiles=conv_tiles,
        tiles_per_page=tiles_per_page,
        tiles=tiles,
        big_page_token_num=big_page_token_num,
        full_att_layer_num=gpu_kv_full_att_state.shape[1],
        BLOCK=BLOCK,
        multibuffer=False,
    )
    for i, page_id in enumerate(valid_pages):
        if write_full_att:
            cpu_cache_full_att[page_id, att_head].view(dtype=torch.uint32).copy_(dest_att[i])
        cpu_cache_conv[page_id, tp_rank].view(dtype=torch.uint32).copy_(dest_conv[i])
        cpu_cache_ssm[page_id, tp_rank].view(dtype=torch.uint32).copy_(dest_ssm[i])


def _launch_copy_cpu_cache_to_kv_npu(
    mem_indexes,
    page_indexes,
    big_page_buffer_ids,
    cpu_cache_full_att,
    cpu_cache_conv,
    cpu_cache_ssm,
    gpu_kv_full_att_state,
    cpu_kv_conv_state,
    cpu_kv_ssm_state,
    tp_rank,
    big_page_token_num,
    head_scale_size,
):
    valid_pages, valid_big, token = _pick_valid_pages(
        page_indexes, None, big_page_buffer_ids, mem_indexes, big_page_token_num
    )
    if not valid_pages:
        return
    att_head = tp_rank // head_scale_size
    device = gpu_kv_full_att_state.device
    gpu_u32, gpu_stride_s, gpu_stride_l, gpu_stride_d, att_u32, conv_u32, ssm_u32 = _npu_copy_meta(
        gpu_kv_full_att_state, cpu_kv_conv_state, cpu_kv_ssm_state, big_page_token_num
    )
    batch = len(valid_pages)
    src_att, dest_conv, dest_ssm, src_conv, src_ssm = _npu_page_staging(batch, att_u32, conv_u32, ssm_u32, device)
    for i, page_id in enumerate(valid_pages):
        src_att[i].copy_(cpu_cache_full_att[page_id, att_head].view(dtype=torch.uint32).reshape(-1))
        src_conv[i].copy_(cpu_cache_conv[page_id, tp_rank].view(dtype=torch.uint32).reshape(-1))
        src_ssm[i].copy_(cpu_cache_ssm[page_id, tp_rank].view(dtype=torch.uint32).reshape(-1))
    BLOCK = 1024
    att_tiles = triton.cdiv(att_u32, BLOCK)
    conv_tiles = triton.cdiv(conv_u32, BLOCK)
    tiles_per_page = att_tiles + conv_tiles + triton.cdiv(ssm_u32, BLOCK)
    tiles = batch * tiles_per_page
    _copy_cpu_cache_to_kv_npu[(min(_npu_vector_core_count(), tiles),)](
        mem_indexes_ptr=token,
        gpu_kv=gpu_u32,
        gpu_stride_s=gpu_stride_s,
        gpu_stride_l=gpu_stride_l,
        gpu_stride_d=gpu_stride_d,
        src_att=src_att,
        src_att_stride_p=src_att.stride(0),
        src_conv=src_conv,
        dest_conv=dest_conv,
        conv_stride_s=src_conv.stride(0),
        dest_conv_stride_p=dest_conv.stride(0),
        src_ssm=src_ssm,
        dest_ssm=dest_ssm,
        ssm_stride_s=src_ssm.stride(0),
        dest_ssm_stride_p=dest_ssm.stride(0),
        att_u32=att_u32,
        conv_u32=conv_u32,
        ssm_u32=ssm_u32,
        att_tiles=att_tiles,
        conv_tiles=conv_tiles,
        tiles_per_page=tiles_per_page,
        tiles=tiles,
        big_page_token_num=big_page_token_num,
        full_att_layer_num=gpu_kv_full_att_state.shape[1],
        BLOCK=BLOCK,
        multibuffer=False,
    )
    for i, big_id in enumerate(valid_big):
        cpu_kv_conv_state[big_id].view(dtype=torch.uint32).reshape(-1).copy_(dest_conv[i])
        cpu_kv_ssm_state[big_id].view(dtype=torch.uint32).reshape(-1).copy_(dest_ssm[i])


@triton.jit
def _copy_kv_buffer_to_cpu_cache(
    page_num,
    mem_indexes_ptr,  # [move_token_num]
    page_indexes_ptr,  # [page_num],
    page_readies_ptr,  # [page_num],
    big_page_buffer_ids,  # [page_num]
    cpu_cache_full_att,  # [all_page_num, head, xdim]
    cpu_cache_full_att_stride_p,
    cpu_cache_full_att_stride_h,
    cpu_cache_full_att_stride_d,
    cpu_cache_conv,  # [all_page_num, tp_world_size, xxdim]
    cpu_cache_conv_stride_p,
    cpu_cache_conv_stride_t,
    cpu_cache_conv_stride_d,
    cpu_cache_ssm,  # [all_page_num, tp_world_size, xxxdim]
    cpu_cache_ssm_stride_p,
    cpu_cache_ssm_stride_t,
    cpu_cache_ssm_stride_d,
    gpu_kv_full_att_state,  # [token_size, full_att_layer_num, xdim]
    gpu_kv_full_att_stride_s,
    gpu_kv_full_att_stride_l,
    gpu_kv_full_att_stride_d,
    cpu_kv_conv_state,  # [buffer_count, xxxxxdim]
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_state,  # [buffer_count, xxxxxxxdim]
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_d,
    gpu_full_att_tail_dim,
    cpu_kv_conv_tail_dim,
    cpu_kv_ssm_tail_dim,
    tp_rank,
    full_att_layer_num,
    big_page_token_num,
    head_scale_size,
    BLOCK: tl.constexpr,
):
    split_index_start = tl.program_id(0)
    grid_num = tl.num_programs(0)
    # 将 所有stride 切成 tl.int64
    cpu_cache_full_att_stride_p = tl.cast(cpu_cache_full_att_stride_p, tl.int64)
    cpu_cache_full_att_stride_h = tl.cast(cpu_cache_full_att_stride_h, tl.int64)
    cpu_cache_full_att_stride_d = tl.cast(cpu_cache_full_att_stride_d, tl.int64)
    cpu_cache_conv_stride_p = tl.cast(cpu_cache_conv_stride_p, tl.int64)
    cpu_cache_conv_stride_t = tl.cast(cpu_cache_conv_stride_t, tl.int64)
    cpu_cache_conv_stride_d = tl.cast(cpu_cache_conv_stride_d, tl.int64)
    cpu_cache_ssm_stride_p = tl.cast(cpu_cache_ssm_stride_p, tl.int64)
    cpu_cache_ssm_stride_t = tl.cast(cpu_cache_ssm_stride_t, tl.int64)
    cpu_cache_ssm_stride_d = tl.cast(cpu_cache_ssm_stride_d, tl.int64)
    gpu_kv_full_att_stride_s = tl.cast(gpu_kv_full_att_stride_s, tl.int64)
    gpu_kv_full_att_stride_l = tl.cast(gpu_kv_full_att_stride_l, tl.int64)
    gpu_kv_full_att_stride_d = tl.cast(gpu_kv_full_att_stride_d, tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, tl.int64)
    cpu_kv_conv_stride_d = tl.cast(cpu_kv_conv_stride_d, tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, tl.int64)
    cpu_kv_ssm_stride_d = tl.cast(cpu_kv_ssm_stride_d, tl.int64)

    for block_index in range(page_num):
        cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)
        run_flag = 1
        if cpu_page_index == -1:
            run_flag = 0
        ready_state = tl.load(page_readies_ptr + block_index)
        if ready_state:
            run_flag = 0
        if tp_rank % head_scale_size == 0:
            head_flag = 1
        else:
            head_flag = 0

        mem_start_ptr = mem_indexes_ptr + big_page_token_num * block_index
        for i in range(split_index_start, tl.cdiv(gpu_full_att_tail_dim, BLOCK) * run_flag * head_flag, grid_num):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < gpu_full_att_tail_dim
            per_token_size = gpu_full_att_tail_dim // big_page_token_num
            per_layer_size = per_token_size // full_att_layer_num
            mem_offs = gpu_start_i // (per_token_size)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-1)
            layer_index = (gpu_start_i // (per_layer_size)) % full_att_layer_num
            dim_index = gpu_start_i % per_layer_size
            gpu_full_att_data = tl.load(
                gpu_kv_full_att_state
                + mem_index * gpu_kv_full_att_stride_s
                + layer_index * gpu_kv_full_att_stride_l
                + dim_index * gpu_kv_full_att_stride_d,
                mask=mask & (mem_index != -1),
                other=0,
            )
            dest_cpu_cache_full_att_ptr = (
                cpu_cache_full_att
                + cpu_page_index * cpu_cache_full_att_stride_p
                + (tp_rank // head_scale_size) * cpu_cache_full_att_stride_h
                + gpu_start_i
            )
            tl.store(dest_cpu_cache_full_att_ptr, gpu_full_att_data, mask=mask & (mem_index != -1))

        big_page_idx = tl.load(big_page_buffer_ids + block_index)

        for i in range(split_index_start, tl.cdiv(cpu_kv_conv_tail_dim, BLOCK) * run_flag, grid_num):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < cpu_kv_conv_tail_dim
            cpu_kv_conv_data = tl.load(
                cpu_kv_conv_state + big_page_idx * cpu_kv_conv_stride_s + gpu_start_i,
                mask=mask,
                other=0,
            )
            dest_cpu_cache_conv_ptr = (
                cpu_cache_conv
                + cpu_page_index * cpu_cache_conv_stride_p
                + tp_rank * cpu_cache_conv_stride_t
                + gpu_start_i
            )
            tl.store(dest_cpu_cache_conv_ptr, cpu_kv_conv_data, mask=mask)

        for i in range(split_index_start, tl.cdiv(cpu_kv_ssm_tail_dim, BLOCK) * run_flag, grid_num):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < cpu_kv_ssm_tail_dim

            cpu_kv_ssm_data = tl.load(
                cpu_kv_ssm_state + big_page_idx * cpu_kv_ssm_stride_s + gpu_start_i,
                mask=mask,
                other=0,
            )
            dest_cpu_cache_ssm_ptr = (
                cpu_cache_ssm + cpu_page_index * cpu_cache_ssm_stride_p + tp_rank * cpu_cache_ssm_stride_t + gpu_start_i
            )
            tl.store(dest_cpu_cache_ssm_ptr, cpu_kv_ssm_data, mask=mask)

    return


def copy_kv_buffer_to_cpu_cache(
    mem_indexes: torch.Tensor,
    page_indexes: torch.Tensor,
    page_readies: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    gpu_kv_full_att_state: torch.Tensor,  # [full_att_layer_num, s, head_num, head_dim]
    cpu_kv_conv_state: torch.Tensor,  # [s, linear_layer_num, dim]
    cpu_kv_ssm_state: torch.Tensor,  # [s, linear_layer_num, xdim]
    cpu_cache_tensor: torch.Tensor,  # [page_num, 1, 1, 1, xxdim]
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    linear_config: LinearAttCacheConfig,
    grid_num: int = 12,
):
    assert len(page_indexes) == len(page_readies) == len(big_page_buffer_ids)
    assert len(mem_indexes) % len(page_indexes) == 0

    BLOCK = 4096
    if linear_config.full_att_all_num_kv_heads % tp_world_size == 0:
        # tp world size 不比 kv 的 head 多时
        head_scale_size = 1
    else:
        head_scale_size = tp_world_size // linear_config.full_att_all_num_kv_heads

    cpu_page_num = cpu_cache_tensor.shape[0]
    cpu_cache_tensor = cpu_cache_tensor.view(cpu_page_num, -1).view(dtype=torch.uint8)
    a = linear_config.get_cpu_cache_full_att_bytes()
    b = linear_config.get_cpu_cache_conv_bytes()
    c = linear_config.get_cpu_cache_ssm_bytes()

    if head_scale_size == 1:
        cpu_cache_full_att = cpu_cache_tensor[:, 0:a].view(cpu_page_num, tp_world_size, -1)
    else:
        cpu_cache_full_att = cpu_cache_tensor[:, 0:a].view(cpu_page_num, linear_config.full_att_all_num_kv_heads, -1)

    cpu_cache_full_att = cpu_cache_full_att.view(dtype=torch.uint64)

    cpu_cache_conv = cpu_cache_tensor[:, a : (a + b)].view(cpu_page_num, tp_world_size, -1).view(dtype=torch.uint64)
    cpu_cache_ssm = (
        cpu_cache_tensor[:, (a + b) : (a + b + c)].view(cpu_page_num, tp_world_size, -1).view(dtype=torch.uint64)
    )

    gpu_kv_full_att_state = gpu_kv_full_att_state.view(
        gpu_kv_full_att_state.shape[0], gpu_kv_full_att_state.shape[1], -1
    ).view(dtype=torch.uint64)

    gpu_kv_full_att_state = gpu_kv_full_att_state.permute(1, 0, 2)  # [s, layer_num, xxdim]

    cpu_kv_conv_state = cpu_kv_conv_state.view(cpu_kv_conv_state.shape[0], -1).view(dtype=torch.uint64)
    cpu_kv_ssm_state = cpu_kv_ssm_state.view(cpu_kv_ssm_state.shape[0], -1).view(dtype=torch.uint64)

    gpu_full_att_tail_dim = gpu_kv_full_att_state.shape[-1] * gpu_kv_full_att_state.shape[-2] * big_page_token_num
    cpu_kv_conv_tail_dim = cpu_kv_conv_state.shape[-1]
    cpu_kv_ssm_tail_dim = cpu_kv_ssm_state.shape[-1]
    full_att_layer_num = gpu_kv_full_att_state.shape[-2]

    assert full_att_layer_num == linear_config.get_full_att_kv_layer_num_with_draft_model()
    assert gpu_full_att_tail_dim == cpu_cache_full_att.shape[-1]
    assert cpu_cache_conv.shape[-1] == cpu_kv_conv_state.shape[-1]
    assert cpu_cache_ssm.shape[-1] == cpu_kv_ssm_state.shape[-1]
    assert gpu_kv_full_att_state.stride(2) == 1
    assert (
        gpu_full_att_tail_dim % big_page_token_num == 0
        and (gpu_full_att_tail_dim // big_page_token_num) % full_att_layer_num == 0
    )
    assert (tp_rank // head_scale_size) < linear_config.full_att_all_num_kv_heads

    if gpu_kv_full_att_state.device.type == "npu":
        _launch_copy_kv_to_cpu_cache_npu(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            page_readies=page_readies,
            big_page_buffer_ids=big_page_buffer_ids,
            cpu_cache_full_att=cpu_cache_full_att,
            cpu_cache_conv=cpu_cache_conv,
            cpu_cache_ssm=cpu_cache_ssm,
            gpu_kv_full_att_state=gpu_kv_full_att_state,
            cpu_kv_conv_state=cpu_kv_conv_state,
            cpu_kv_ssm_state=cpu_kv_ssm_state,
            tp_rank=tp_rank,
            big_page_token_num=big_page_token_num,
            head_scale_size=head_scale_size,
        )
        return

    grid = (grid_num,)
    _copy_kv_buffer_to_cpu_cache[grid](
        page_num=len(page_indexes),
        mem_indexes_ptr=mem_indexes,
        page_indexes_ptr=page_indexes,
        page_readies_ptr=page_readies,
        big_page_buffer_ids=big_page_buffer_ids,
        cpu_cache_full_att=cpu_cache_full_att,
        cpu_cache_full_att_stride_p=cpu_cache_full_att.stride(0),
        cpu_cache_full_att_stride_h=cpu_cache_full_att.stride(1),
        cpu_cache_full_att_stride_d=cpu_cache_full_att.stride(2),
        cpu_cache_conv=cpu_cache_conv,
        cpu_cache_conv_stride_p=cpu_cache_conv.stride(0),
        cpu_cache_conv_stride_t=cpu_cache_conv.stride(1),
        cpu_cache_conv_stride_d=cpu_cache_conv.stride(2),
        cpu_cache_ssm=cpu_cache_ssm,
        cpu_cache_ssm_stride_p=cpu_cache_ssm.stride(0),
        cpu_cache_ssm_stride_t=cpu_cache_ssm.stride(1),
        cpu_cache_ssm_stride_d=cpu_cache_ssm.stride(2),
        gpu_kv_full_att_state=gpu_kv_full_att_state,
        gpu_kv_full_att_stride_s=gpu_kv_full_att_state.stride(0),
        gpu_kv_full_att_stride_l=gpu_kv_full_att_state.stride(1),
        gpu_kv_full_att_stride_d=gpu_kv_full_att_state.stride(2),
        cpu_kv_conv_state=cpu_kv_conv_state,
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(1),
        cpu_kv_ssm_state=cpu_kv_ssm_state,
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(1),
        gpu_full_att_tail_dim=gpu_full_att_tail_dim,
        cpu_kv_conv_tail_dim=cpu_kv_conv_tail_dim,
        cpu_kv_ssm_tail_dim=cpu_kv_ssm_tail_dim,
        tp_rank=tp_rank,
        full_att_layer_num=full_att_layer_num,
        big_page_token_num=big_page_token_num,
        head_scale_size=head_scale_size,
        BLOCK=BLOCK,
    )


@triton.jit
def _copy_cpu_cache_to_kv_buffer(
    page_num,
    mem_indexes_ptr,  # [move_token_num]
    page_indexes_ptr,  # [page_num],
    big_page_buffer_ids,  # [page_num]
    cpu_cache_full_att,  # [all_page_num, head, xdim]
    cpu_cache_full_att_stride_p,
    cpu_cache_full_att_stride_h,
    cpu_cache_full_att_stride_d,
    cpu_cache_conv,  # [all_page_num, tp_world_size, xxdim]
    cpu_cache_conv_stride_p,
    cpu_cache_conv_stride_t,
    cpu_cache_conv_stride_d,
    cpu_cache_ssm,  # [all_page_num, tp_world_size, xxxdim]
    cpu_cache_ssm_stride_p,
    cpu_cache_ssm_stride_t,
    cpu_cache_ssm_stride_d,
    gpu_kv_full_att_state,  # [token_size, full_att_layer_num, xdim]
    gpu_kv_full_att_stride_s,
    gpu_kv_full_att_stride_l,
    gpu_kv_full_att_stride_d,
    cpu_kv_conv_state,  # [buffer_count, xxxxxdim]
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_state,  # [buffer_count, xxxxxxxdim]
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_d,
    gpu_full_att_tail_dim,
    cpu_kv_conv_tail_dim,
    cpu_kv_ssm_tail_dim,
    tp_rank,
    full_att_layer_num,
    big_page_token_num,
    head_scale_size,
    BLOCK: tl.constexpr,
):
    split_index_start = tl.program_id(0)
    grid_num = tl.num_programs(0)
    # 将 所有stride 切成 tl.int64
    cpu_cache_full_att_stride_p = tl.cast(cpu_cache_full_att_stride_p, tl.int64)
    cpu_cache_full_att_stride_h = tl.cast(cpu_cache_full_att_stride_h, tl.int64)
    cpu_cache_full_att_stride_d = tl.cast(cpu_cache_full_att_stride_d, tl.int64)
    cpu_cache_conv_stride_p = tl.cast(cpu_cache_conv_stride_p, tl.int64)
    cpu_cache_conv_stride_t = tl.cast(cpu_cache_conv_stride_t, tl.int64)
    cpu_cache_conv_stride_d = tl.cast(cpu_cache_conv_stride_d, tl.int64)
    cpu_cache_ssm_stride_p = tl.cast(cpu_cache_ssm_stride_p, tl.int64)
    cpu_cache_ssm_stride_t = tl.cast(cpu_cache_ssm_stride_t, tl.int64)
    cpu_cache_ssm_stride_d = tl.cast(cpu_cache_ssm_stride_d, tl.int64)
    gpu_kv_full_att_stride_s = tl.cast(gpu_kv_full_att_stride_s, tl.int64)
    gpu_kv_full_att_stride_l = tl.cast(gpu_kv_full_att_stride_l, tl.int64)
    gpu_kv_full_att_stride_d = tl.cast(gpu_kv_full_att_stride_d, tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, tl.int64)
    cpu_kv_conv_stride_d = tl.cast(cpu_kv_conv_stride_d, tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, tl.int64)
    cpu_kv_ssm_stride_d = tl.cast(cpu_kv_ssm_stride_d, tl.int64)

    for block_index in range(page_num):
        cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)

        mem_start_ptr = mem_indexes_ptr + big_page_token_num * block_index
        for i in range(split_index_start, tl.cdiv(gpu_full_att_tail_dim, BLOCK), grid_num):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < gpu_full_att_tail_dim
            per_token_size = gpu_full_att_tail_dim // big_page_token_num
            per_layer_size = per_token_size // full_att_layer_num
            mem_offs = gpu_start_i // (per_token_size)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-1)
            layer_index = (gpu_start_i // (per_layer_size)) % full_att_layer_num
            dim_index = gpu_start_i % per_layer_size

            src_cpu_cache_full_att_ptr = (
                cpu_cache_full_att
                + cpu_page_index * cpu_cache_full_att_stride_p
                + (tp_rank // head_scale_size) * cpu_cache_full_att_stride_h
                + gpu_start_i
            )
            cpu_full_att_data = tl.load(src_cpu_cache_full_att_ptr, mask=mask & (mem_index != -1), other=0)

            tl.store(
                gpu_kv_full_att_state
                + mem_index * gpu_kv_full_att_stride_s
                + layer_index * gpu_kv_full_att_stride_l
                + dim_index * gpu_kv_full_att_stride_d,
                cpu_full_att_data,
                mask=mask & (mem_index != -1),
            )

        big_page_idx = tl.load(big_page_buffer_ids + block_index)

        for i in range(split_index_start, tl.cdiv(cpu_kv_conv_tail_dim, BLOCK), grid_num):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < cpu_kv_conv_tail_dim

            src_cpu_cache_conv_ptr = (
                cpu_cache_conv
                + cpu_page_index * cpu_cache_conv_stride_p
                + tp_rank * cpu_cache_conv_stride_t
                + gpu_start_i
            )
            cpu_kv_conv_data = tl.load(src_cpu_cache_conv_ptr, mask=mask, other=0)

            tl.store(
                cpu_kv_conv_state + big_page_idx * cpu_kv_conv_stride_s + gpu_start_i,
                cpu_kv_conv_data,
                mask=mask,
            )

        for i in range(split_index_start, tl.cdiv(cpu_kv_ssm_tail_dim, BLOCK), grid_num):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < cpu_kv_ssm_tail_dim

            src_cpu_cache_ssm_ptr = (
                cpu_cache_ssm + cpu_page_index * cpu_cache_ssm_stride_p + tp_rank * cpu_cache_ssm_stride_t + gpu_start_i
            )
            cpu_kv_ssm_data = tl.load(src_cpu_cache_ssm_ptr, mask=mask, other=0)

            tl.store(
                cpu_kv_ssm_state + big_page_idx * cpu_kv_ssm_stride_s + gpu_start_i,
                cpu_kv_ssm_data,
                mask=mask,
            )

    return


def copy_cpu_cache_to_kv_buffer(
    mem_indexes: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    page_indexes: torch.Tensor,
    gpu_full_att_kv_state: torch.Tensor,  # [layer_num, s, head_num, head_dim]
    cpu_kv_conv_state: torch.Tensor,  # [layer_num, s, dim]
    cpu_kv_ssm_state: torch.Tensor,  # [layer_num, s, xdim]
    cpu_cache_tensor: torch.Tensor,  # [page_num, 1, 1, tp_world_size, xxdim]
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    linear_config: LinearAttCacheConfig,
    grid_num: int = 12,
):
    assert len(mem_indexes) % len(page_indexes) == 0

    BLOCK = 4096
    if linear_config.full_att_all_num_kv_heads % tp_world_size == 0:
        head_scale_size = 1
    else:
        head_scale_size = tp_world_size // linear_config.full_att_all_num_kv_heads

    cpu_page_num = cpu_cache_tensor.shape[0]
    cpu_cache_tensor = cpu_cache_tensor.view(cpu_page_num, -1).view(dtype=torch.uint8)
    a = linear_config.get_cpu_cache_full_att_bytes()
    b = linear_config.get_cpu_cache_conv_bytes()
    c = linear_config.get_cpu_cache_ssm_bytes()

    if head_scale_size == 1:
        cpu_cache_full_att = cpu_cache_tensor[:, 0:a].view(cpu_page_num, tp_world_size, -1)
    else:
        cpu_cache_full_att = cpu_cache_tensor[:, 0:a].view(cpu_page_num, linear_config.full_att_all_num_kv_heads, -1)

    cpu_cache_full_att = cpu_cache_full_att.view(dtype=torch.uint64)

    cpu_cache_conv = cpu_cache_tensor[:, a : (a + b)].view(cpu_page_num, tp_world_size, -1).view(dtype=torch.uint64)
    cpu_cache_ssm = (
        cpu_cache_tensor[:, (a + b) : (a + b + c)].view(cpu_page_num, tp_world_size, -1).view(dtype=torch.uint64)
    )

    gpu_full_att_kv_state = gpu_full_att_kv_state.view(
        gpu_full_att_kv_state.shape[0], gpu_full_att_kv_state.shape[1], -1
    ).view(dtype=torch.uint64)
    gpu_full_att_kv_state = gpu_full_att_kv_state.permute(1, 0, 2)  # [s, layer_num, xxdim]

    cpu_kv_conv_state = cpu_kv_conv_state.view(cpu_kv_conv_state.shape[0], -1).view(dtype=torch.uint64)
    cpu_kv_ssm_state = cpu_kv_ssm_state.view(cpu_kv_ssm_state.shape[0], -1).view(dtype=torch.uint64)

    gpu_full_att_tail_dim = gpu_full_att_kv_state.shape[-1] * gpu_full_att_kv_state.shape[-2] * big_page_token_num
    cpu_kv_conv_tail_dim = cpu_kv_conv_state.shape[-1]
    cpu_kv_ssm_tail_dim = cpu_kv_ssm_state.shape[-1]
    full_att_layer_num = gpu_full_att_kv_state.shape[-2]

    assert gpu_full_att_tail_dim == cpu_cache_full_att.shape[-1]
    assert cpu_cache_conv.shape[-1] == cpu_kv_conv_state.shape[-1]
    assert cpu_cache_ssm.shape[-1] == cpu_kv_ssm_state.shape[-1]
    assert gpu_full_att_kv_state.stride(2) == 1

    assert (tp_rank // head_scale_size) < linear_config.full_att_all_num_kv_heads

    if gpu_full_att_kv_state.device.type == "npu":
        _launch_copy_cpu_cache_to_kv_npu(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            big_page_buffer_ids=big_page_buffer_ids,
            cpu_cache_full_att=cpu_cache_full_att,
            cpu_cache_conv=cpu_cache_conv,
            cpu_cache_ssm=cpu_cache_ssm,
            gpu_kv_full_att_state=gpu_full_att_kv_state,
            cpu_kv_conv_state=cpu_kv_conv_state,
            cpu_kv_ssm_state=cpu_kv_ssm_state,
            tp_rank=tp_rank,
            big_page_token_num=big_page_token_num,
            head_scale_size=head_scale_size,
        )
        return

    grid = (grid_num,)
    _copy_cpu_cache_to_kv_buffer[grid](
        page_num=len(page_indexes),
        mem_indexes_ptr=mem_indexes,
        page_indexes_ptr=page_indexes,
        big_page_buffer_ids=big_page_buffer_ids,
        cpu_cache_full_att=cpu_cache_full_att,
        cpu_cache_full_att_stride_p=cpu_cache_full_att.stride(0),
        cpu_cache_full_att_stride_h=cpu_cache_full_att.stride(1),
        cpu_cache_full_att_stride_d=cpu_cache_full_att.stride(2),
        cpu_cache_conv=cpu_cache_conv,
        cpu_cache_conv_stride_p=cpu_cache_conv.stride(0),
        cpu_cache_conv_stride_t=cpu_cache_conv.stride(1),
        cpu_cache_conv_stride_d=cpu_cache_conv.stride(2),
        cpu_cache_ssm=cpu_cache_ssm,
        cpu_cache_ssm_stride_p=cpu_cache_ssm.stride(0),
        cpu_cache_ssm_stride_t=cpu_cache_ssm.stride(1),
        cpu_cache_ssm_stride_d=cpu_cache_ssm.stride(2),
        gpu_kv_full_att_state=gpu_full_att_kv_state,
        gpu_kv_full_att_stride_s=gpu_full_att_kv_state.stride(0),
        gpu_kv_full_att_stride_l=gpu_full_att_kv_state.stride(1),
        gpu_kv_full_att_stride_d=gpu_full_att_kv_state.stride(2),
        cpu_kv_conv_state=cpu_kv_conv_state,
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(1),
        cpu_kv_ssm_state=cpu_kv_ssm_state,
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(1),
        gpu_full_att_tail_dim=gpu_full_att_tail_dim,
        cpu_kv_conv_tail_dim=cpu_kv_conv_tail_dim,
        cpu_kv_ssm_tail_dim=cpu_kv_ssm_tail_dim,
        tp_rank=tp_rank,
        full_att_layer_num=full_att_layer_num,
        big_page_token_num=big_page_token_num,
        head_scale_size=head_scale_size,
        BLOCK=BLOCK,
    )


@triton.jit
def _copy_linear_att_state_to_linear_att_state(
    src_conv_state,
    dst_conv_state,
    src_ssm_state,
    dst_ssm_state,
    conv_size,
    ssm_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_num = tl.num_programs(0)

    # copy conv state
    num_conv_blocks = tl.cdiv(conv_size, BLOCK)
    for i in range(pid, num_conv_blocks, grid_num):
        start = i * BLOCK + tl.arange(0, BLOCK)
        mask = start < conv_size
        data = tl.load(src_conv_state + start, mask=mask, other=0)
        tl.store(dst_conv_state + start, data, mask=mask)

    # copy ssm state
    num_ssm_blocks = tl.cdiv(ssm_size, BLOCK)
    for i in range(pid, num_ssm_blocks, grid_num):
        start = i * BLOCK + tl.arange(0, BLOCK)
        mask = start < ssm_size
        data = tl.load(src_ssm_state + start, mask=mask, other=0)
        tl.store(dst_ssm_state + start, data, mask=mask)


def copy_linear_att_state_to_linear_att_state(
    src_conv_state: torch.Tensor,
    src_ssm_state: torch.Tensor,
    dst_conv_state: torch.Tensor,
    dst_ssm_state: torch.Tensor,
    grid_num: int = 16,
):
    assert src_conv_state.shape == dst_conv_state.shape
    assert src_ssm_state.shape == dst_ssm_state.shape
    if src_conv_state.device.type != "cuda":
        dst_conv_state.copy_(src_conv_state)
        dst_ssm_state.copy_(src_ssm_state)
        return

    BLOCK = 4096

    src_conv_flat = src_conv_state.view(-1).view(dtype=torch.uint8)
    dst_conv_flat = dst_conv_state.view(-1).view(dtype=torch.uint8)
    src_ssm_flat = src_ssm_state.view(-1).view(dtype=torch.uint8)
    dst_ssm_flat = dst_ssm_state.view(-1).view(dtype=torch.uint8)

    conv_size = src_conv_flat.shape[0]
    ssm_size = src_ssm_flat.shape[0]

    grid = (grid_num,)
    _copy_linear_att_state_to_linear_att_state[grid](
        src_conv_state=src_conv_flat,
        dst_conv_state=dst_conv_flat,
        src_ssm_state=src_ssm_flat,
        dst_ssm_state=dst_ssm_flat,
        conv_size=conv_size,
        ssm_size=ssm_size,
        BLOCK=BLOCK,
    )
