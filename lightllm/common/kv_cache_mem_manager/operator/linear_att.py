import torch
import triton
from typing import List
from typing import TYPE_CHECKING
from .base import BaseMemManagerOperator
from lightllm.utils.envs_utils import get_env_start_args, get_page_size
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.utils.log_utils import init_logger
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

if TYPE_CHECKING:
    from lightllm.server.multi_level_kv_cache.cpu_cache_client import CpuKvCacheClient
    from lightllm.server.router.model_infer.infer_batch import InferReq

logger = init_logger(__name__)


class LinearAttMemOperator(BaseMemManagerOperator):
    """
    只用于非量化的linear att 混合 full att的模型，列入 qwen3.5
    """

    def __init__(self, mem_manager):
        super().__init__(mem_manager)
        self.linear_config = LinearAttCacheConfig.load_from_args()

    def load_cpu_cache_to_gpu(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ):
        assert mem_indexes.device == self.mem_manager.target_device
        assert page_indexes.device == self.mem_manager.target_device
        args = get_env_start_args()
        assert triton.cdiv(len(mem_indexes), args.cpu_cache_token_page_size) == len(page_indexes)
        assert len(mem_indexes) % args.linear_att_hash_page_size == 0
        assert args.cpu_cache_token_page_size == args.linear_att_hash_page_size * args.linear_att_page_block_num
        from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager

        mem_manager: Qwen3NextMemManager = self.mem_manager

        big_page_num = len(mem_indexes) // args.cpu_cache_token_page_size
        max_kv_len = (req.cur_kv_len // args.cpu_cache_token_page_size) * args.cpu_cache_token_page_size
        assert max_kv_len % args.cpu_cache_token_page_size == 0

        big_page_buffer_ids_cpu = []
        for i in range(big_page_num):
            page_id = mem_manager.linear_att_big_page_buffers.alloc_one_state_cache()
            assert page_id is not None
            req.linear_att_len_to_big_page_id[max_kv_len] = page_id
            big_page_buffer_ids_cpu.append(page_id)
            max_kv_len -= args.cpu_cache_token_page_size
            assert max_kv_len % args.cpu_cache_token_page_size == 0

        big_page_buffer_ids_cpu.reverse()

        # 碎页情况的处理
        has_tail_page = len(mem_indexes) % args.cpu_cache_token_page_size != 0
        if has_tail_page:
            padded_token_num = triton.cdiv(
                len(mem_indexes), args.cpu_cache_token_page_size
            ) * args.cpu_cache_token_page_size - len(mem_indexes)
            mem_indexes = torch.nn.functional.pad(mem_indexes, (0, padded_token_num), mode="constant", value=-1)

            # 将对应的小叶数据拷贝到临时的大页上，再从大页上拷贝到对应的运行态页面上
            big_page_buffer_ids_cpu.append(mem_manager.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID)

        big_page_buffer_ids_gpu = torch.tensor(big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu").to(
            device=self.mem_manager.target_device,
            non_blocking=True
        )

        assert len(big_page_buffer_ids_gpu) == len(page_indexes)

        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_cpu_cache_to_kv_buffer,
        )

        copy_cpu_cache_to_kv_buffer(
            mem_indexes=mem_indexes,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            page_indexes=page_indexes,
            gpu_full_att_kv_state=mem_manager.kv_buffer,
            cpu_kv_conv_state=mem_manager.linear_att_big_page_buffers.conv_state_cache.buffer,
            cpu_kv_ssm_state=mem_manager.linear_att_big_page_buffers.ssm_state_cache.buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        g_infer_context.req_manager.copy_big_page_buffer_to_linear_att_state(
            big_page_buffer_idx=big_page_buffer_ids_cpu[-1],
            req=req,
        )

        return

    def offload_gpu_kv_to_cpu_cache(
        self,
        mem_indexes: torch.Tensor,
        page_indexes: torch.Tensor,
        page_readies: torch.Tensor,
        cpu_cache_client: "CpuKvCacheClient",
        req: "InferReq",
    ):
        args = get_env_start_args()
        if not hasattr(self, "big_page_ids_buffer_store"):
            self.big_page_ids_buffer_store = torch.empty((1024 * 1024 * 4,), dtype=torch.int64, device=self.mem_manager.target_device)
            # 多申请3个cpu cache token page size，用于处理碎页情况，碎页情况需要将对应的大页数据拷贝到临时的大页上，
            # 再从大页上拷贝到对应的运行态页面上
            self.mem_indexes_buffer = torch.empty(
                (args.max_req_total_len + 3 * args.cpu_cache_token_page_size,), dtype=torch.int32, device=self.mem_manager.target_device
            )

        assert len(mem_indexes) % args.linear_att_hash_page_size == 0
        assert triton.cdiv(len(mem_indexes), args.cpu_cache_token_page_size) == len(page_indexes)

        from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager

        mem_manager: Qwen3NextMemManager = self.mem_manager

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        big_page_buffer_ids_cpu = g_infer_context.radix_cache.get_big_page_ids_by_node(req.shared_kv_node)
        max_kv_len = (len(mem_indexes) // args.cpu_cache_token_page_size) * args.cpu_cache_token_page_size
        start_kv_len = (len(big_page_buffer_ids_cpu) + 1) * args.cpu_cache_token_page_size
        for seq_len in range(start_kv_len, max_kv_len + 1, args.cpu_cache_token_page_size):
            page_id = req.linear_att_len_to_big_page_id[seq_len]
            big_page_buffer_ids_cpu.append(page_id)

        if len(mem_indexes) % args.cpu_cache_token_page_size != 0:
            # 存在不满大页的碎页的页面存在需要复制的情况
            dst_len = triton.cdiv(len(mem_indexes), args.cpu_cache_token_page_size) * args.cpu_cache_token_page_size
            assert dst_len <= self.mem_indexes_buffer.shape[0]
            dst_mem_indexes = self.mem_indexes_buffer[0:dst_len].fill_(-1)
            dst_mem_indexes[0 : len(mem_indexes)].copy_(mem_indexes, non_blocking=True)
            mem_indexes = dst_mem_indexes
            assert req.tail_linear_att_small_page_buffer_id is not None
            from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
                copy_linear_att_state_to_linear_att_state,
            )

            src_conv_state, src_ssm_state = g_infer_context.radix_cache.linear_att_small_page_buffers.get_state_cache(
                buffer_idx=req.tail_linear_att_small_page_buffer_id
            )
            dst_conv_state, dst_ssm_state = mem_manager.linear_att_big_page_buffers.get_state_cache(
                buffer_idx=mem_manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID,
            )
            copy_linear_att_state_to_linear_att_state(
                src_conv_state=src_conv_state,
                src_ssm_state=src_ssm_state,
                dst_conv_state=dst_conv_state,
                dst_ssm_state=dst_ssm_state,
            )
            big_page_buffer_ids_cpu.append(mem_manager.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID)

        assert len(big_page_buffer_ids_cpu) == len(page_indexes) == len(page_readies)

        big_page_buffer_ids_cpu = torch.tensor(
            big_page_buffer_ids_cpu, dtype=torch.int64, device="cpu", pin_memory=True
        )
        assert len(big_page_buffer_ids_cpu) <= self.big_page_ids_buffer_store.shape[0]
        big_page_buffer_ids_gpu = self.big_page_ids_buffer_store[0 : len(big_page_buffer_ids_cpu)]
        big_page_buffer_ids_gpu.copy_(big_page_buffer_ids_cpu, non_blocking=True)

        from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
            copy_kv_buffer_to_cpu_cache,
        )

        copy_kv_buffer_to_cpu_cache(
            mem_indexes=mem_indexes,
            page_indexes=page_indexes,
            page_readies=page_readies,
            big_page_buffer_ids=big_page_buffer_ids_gpu,
            gpu_kv_full_att_state=mem_manager.kv_buffer,
            cpu_kv_conv_state=mem_manager.linear_att_big_page_buffers.conv_state_cache.buffer,
            cpu_kv_ssm_state=mem_manager.linear_att_big_page_buffers.ssm_state_cache.buffer,
            cpu_cache_tensor=cpu_cache_client.cpu_kv_cache_tensor,
            tp_rank=get_current_rank_in_dp(),
            tp_world_size=get_dp_world_size(),
            big_page_token_num=args.cpu_cache_token_page_size,
            linear_config=self.linear_config,
        )
        return

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        # Qwen3Next 需要调整 layer_index
        layer_index = self.linear_config.get_full_att_kv_layer_index(layer_index)
        from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager

        mem_manager: MemoryManager = self.mem_manager
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import (
            destindex_copy_kv,
        )

        destindex_copy_kv(kv, mem_index, mem_manager.kv_buffer[layer_index])
        return

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        copy_kv_buffer_to_kv_buffer(
            src_mem_index.to(device=self.mem_manager.target_device, non_blocking=True),
            dst_mem_index.to(device=self.mem_manager.target_device, non_blocking=True),
            self.mem_manager.kv_buffer,
        )
        return


class Int8KVLinearAttMemOperator(LinearAttMemOperator):

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        mem_manager = self.mem_manager
        layer_index = mem_manager.get_full_att_cache_layer_index(layer_index)
        if kv.shape[0] == 0:
            return
        from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import (
            destindex_copy_quantize_kv,
        )

        destindex_copy_quantize_kv(
            kv,
            mem_index,
            mem_manager.kv_buffer[layer_index],
            mem_manager.scale_buffer[layer_index],
            quant_group_dim=mem_manager.group_quant_size,
        )
        return

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        src_mem_index = src_mem_index.to(device=self.mem_manager.target_device, non_blocking=True)
        dst_mem_index = dst_mem_index.to(device=self.mem_manager.target_device, non_blocking=True)
        copy_kv_buffer_to_kv_buffer(src_mem_index, dst_mem_index, self.mem_manager.kv_buffer)
        copy_kv_buffer_to_kv_buffer(src_mem_index, dst_mem_index, self.mem_manager.scale_buffer)
        return

    def load_cpu_cache_to_gpu(self, *args, **kwargs):
        raise NotImplementedError("CPU KV cache is not implemented for Qwen3.6 INT8 KV")

    def offload_gpu_kv_to_cpu_cache(self, *args, **kwargs):
        raise NotImplementedError("CPU KV cache is not implemented for Qwen3.6 INT8 KV")


class NpuLinearAttMemOperator(LinearAttMemOperator):

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        mem_manager = self.mem_manager
        layer_index = mem_manager.get_full_att_cache_layer_index(layer_index)
        k_src = kv[:, : mem_manager.head_num, :]
        v_src = kv[:, mem_manager.head_num :, :]

        import torch_npu

        page_size = get_page_size()
        num_blocks = mem_manager.k_buffer.shape[1] // page_size
        key_cache = mem_manager.k_buffer[layer_index].view(
            num_blocks, page_size, mem_manager.head_num, mem_manager.head_dim
        )
        value_cache = mem_manager.v_buffer[layer_index].view(
            num_blocks, page_size, mem_manager.head_num, mem_manager.head_dim
        )
        slot = mem_index if mem_index.dtype == torch.int32 else mem_index.to(torch.int32)
        torch_npu._npu_reshape_and_cache(
            key=k_src,
            value=v_src,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=slot,
        )

    def copy_mem_to_mem(self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor):
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        src_mem_index = src_mem_index.to(device=self.mem_manager.target_device, non_blocking=True)
        dst_mem_index = dst_mem_index.to(device=self.mem_manager.target_device, non_blocking=True)
        copy_kv_buffer_to_kv_buffer(src_mem_index, dst_mem_index, self.mem_manager.k_buffer)
        copy_kv_buffer_to_kv_buffer(src_mem_index, dst_mem_index, self.mem_manager.v_buffer)

    def _is_int8_split_kv(self) -> bool:
        return getattr(self.mem_manager, "k_scale_buffer", None) is not None

    def _flat_token_kv(self, buf: torch.Tensor) -> torch.Tensor:
        return buf.view(buf.shape[0], -1, buf.shape[-2], buf.shape[-1])

    def _flat_token_scale(self, buf: torch.Tensor) -> torch.Tensor:
        return buf.view(buf.shape[0], -1)

    def _ensure_cuda_layout_staging(self, token_num: int) -> torch.Tensor:
        k_flat = self._flat_token_kv(self.mem_manager.k_buffer)
        layers, _, head_num, head_dim = k_flat.shape
        dtype = self.linear_config.full_att_dtype
        buf = getattr(self, "_cpu_cache_kv_staging", None)
        if buf is None or buf.shape[1] < token_num or buf.dtype != dtype or buf.shape[0] != layers:
            self._cpu_cache_kv_staging = torch.empty(
                (layers, token_num, 2 * head_num, head_dim),
                dtype=dtype,
                device=self.mem_manager.target_device,
            )
        return self._cpu_cache_kv_staging

    def _remap_mem_indexes(self, mem_indexes: torch.Tensor):
        n = mem_indexes.numel()
        valid = mem_indexes >= 0
        remap = torch.arange(n, device=mem_indexes.device, dtype=mem_indexes.dtype)
        remap = torch.where(valid, remap, mem_indexes.new_full((), -1))
        return valid, remap

    def _gather_split_kv_to_cuda_layout(self, mem_indexes: torch.Tensor):
        n = mem_indexes.numel()
        buf = self._ensure_cuda_layout_staging(n)
        packed = buf[:, :n]
        valid, remap = self._remap_mem_indexes(mem_indexes)
        idx = mem_indexes.clamp(min=0).to(torch.int64)
        head_num = self.mem_manager.head_num
        k = self._flat_token_kv(self.mem_manager.k_buffer).index_select(1, idx)
        v = self._flat_token_kv(self.mem_manager.v_buffer).index_select(1, idx)
        if self._is_int8_split_kv():
            k_scale = self._flat_token_scale(self.mem_manager.k_scale_buffer).index_select(1, idx)
            v_scale = self._flat_token_scale(self.mem_manager.v_scale_buffer).index_select(1, idx)
            dest_dtype = packed.dtype
            k = (k.float() * k_scale.unsqueeze(-1).unsqueeze(-1)).to(dest_dtype)
            v = (v.float() * v_scale.unsqueeze(-1).unsqueeze(-1)).to(dest_dtype)
        packed[:, :, :head_num].copy_(k)
        packed[:, :, head_num:].copy_(v)
        packed.masked_fill_(~valid.view(1, -1, 1, 1), 0)
        return packed, remap

    def _scatter_cuda_layout_to_split_kv(self, mem_indexes: torch.Tensor, packed: torch.Tensor):
        valid = mem_indexes >= 0
        valid_idx = mem_indexes.to(torch.int64)[valid]
        if valid_idx.numel() == 0:
            return
        head_num = self.mem_manager.head_num
        k_src = packed[:, valid, :head_num]
        v_src = packed[:, valid, head_num:]
        k_flat = self._flat_token_kv(self.mem_manager.k_buffer)
        v_flat = self._flat_token_kv(self.mem_manager.v_buffer)
        if self._is_int8_split_kv():
            import torch_npu

            layers, token_num, heads, head_dim = k_src.shape
            k_q, k_scale = torch_npu.npu_dynamic_quant(
                k_src.reshape(layers * token_num, -1), dst_type=torch.int8
            )
            v_q, v_scale = torch_npu.npu_dynamic_quant(
                v_src.reshape(layers * token_num, -1), dst_type=torch.int8
            )
            k_flat.index_copy_(1, valid_idx, k_q.view(layers, token_num, heads, head_dim))
            v_flat.index_copy_(1, valid_idx, v_q.view(layers, token_num, heads, head_dim))
            self._flat_token_scale(self.mem_manager.k_scale_buffer).index_copy_(
                1, valid_idx, k_scale.view(layers, token_num)
            )
            self._flat_token_scale(self.mem_manager.v_scale_buffer).index_copy_(
                1, valid_idx, v_scale.view(layers, token_num)
            )
            return
        k_flat.index_copy_(1, valid_idx, k_src)
        v_flat.index_copy_(1, valid_idx, v_src)

    def load_cpu_cache_to_gpu(self, mem_indexes, page_indexes, cpu_cache_client, req):
        n = mem_indexes.numel()
        buf = self._ensure_cuda_layout_staging(n)
        _, remap = self._remap_mem_indexes(mem_indexes)
        old_kv = self.mem_manager.kv_buffer
        self.mem_manager.kv_buffer = buf
        try:
            super().load_cpu_cache_to_gpu(remap, page_indexes, cpu_cache_client, req)
        finally:
            self.mem_manager.kv_buffer = old_kv
        self._scatter_cuda_layout_to_split_kv(mem_indexes, buf[:, :n])

    def offload_gpu_kv_to_cpu_cache(self, mem_indexes, page_indexes, page_readies, cpu_cache_client, req):
        _, remap = self._gather_split_kv_to_cuda_layout(mem_indexes)
        old_kv = self.mem_manager.kv_buffer
        self.mem_manager.kv_buffer = self._cpu_cache_kv_staging
        try:
            return super().offload_gpu_kv_to_cpu_cache(remap, page_indexes, page_readies, cpu_cache_client, req)
        finally:
            self.mem_manager.kv_buffer = old_kv
