import torch
import triton
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.utils.envs_utils import get_env_start_args, get_page_size
from lightllm.common.linear_att_cache_manager import LinearAttCacheConfig, LinearAttCacheManager
from .operator import LinearAttMemOperator
from .operator.linear_att import (
    CoreXLinearAttMemOperator,
    Int8KVLinearAttMemOperator,
    NpuLinearAttMemOperator,
)
from .npu_mem_manager import NPUInt8KVMemoryManager, NPUInt8KVOperator
from lightllm.utils.device_utils import get_target_device, is_maca
from typing import Tuple, Any, List

logger = init_logger(__name__)


class NpuInt8KVLinearAttMemOperator(NpuLinearAttMemOperator):

    def copy_kv_to_mem_manager(
        self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor
    ) -> None:
        layer_index = self.mem_manager.get_full_att_cache_layer_index(layer_index)
        NPUInt8KVOperator.copy_kv_to_mem_manager(self, layer_index, mem_index, kv)

    def copy_mem_to_mem(
        self, src_mem_index: torch.Tensor, dst_mem_index: torch.Tensor
    ) -> None:
        src_mem_index = src_mem_index.to(
            device=self.mem_manager.target_device, dtype=torch.int64, non_blocking=True
        )
        dst_mem_index = dst_mem_index.to(
            device=self.mem_manager.target_device, dtype=torch.int64, non_blocking=True
        )
        buffers = (
            self.mem_manager.k_buffer,
            self.mem_manager.v_buffer,
            self.mem_manager.k_scale_buffer,
            self.mem_manager.v_scale_buffer,
        )
        for buffer in buffers:
            flat_buffer = buffer.flatten(1, 2)
            flat_buffer.index_copy_(1, dst_mem_index, flat_buffer.index_select(1, src_mem_index))


class Qwen3NextMemManager(MemoryManager):
    operator_class = LinearAttMemOperator
    npu_operator_class = NpuLinearAttMemOperator

    def __init__(
        self,
        size,
        dtype,
        num_kv_heads,
        head_dim,
        full_att_layer_num,
        linear_config: LinearAttCacheConfig,
        always_copy=False,
        mem_fraction=0.9,
    ):
        self.linear_config = linear_config

        super().__init__(size, dtype, num_kv_heads, head_dim, full_att_layer_num, always_copy, mem_fraction)
        if self._split_full_att_kv:
            self.operator = self.npu_operator_class(self)

    def get_full_att_cache_layer_index(self, layer_index: int) -> int:
        return self.linear_config.get_full_att_kv_layer_index(layer_index)

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        layer_index = self.get_full_att_cache_layer_index(layer_index)
        if self._split_full_att_kv:
            return self.k_buffer[layer_index], self.v_buffer[layer_index]
        return super().get_att_input_params(layer_index)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self._split_full_att_kv = torch.device(self.target_device).type == "npu"
        if self._split_full_att_kv:
            page_size = get_page_size()
            alloc_size = ((size // page_size) + 1) * page_size if page_size > 1 else size + 1
            shape = (layer_num, alloc_size, head_num, head_dim)
            self.k_buffer = torch.empty(shape, dtype=dtype, device=self.target_device)
            self.v_buffer = torch.empty(shape, dtype=dtype, device=self.target_device)
            self.kv_buffer = None
        else:
            super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        # TODO 初始化线性 att 对应的部分 buffer.
        self._init_linear_att_buffers()
        return

    def get_index_kv_buffer(self, index):
        if not self._split_full_att_kv:
            return super().get_index_kv_buffer(index)
        return {"kv_buffer": torch.cat([self.k_buffer[:, index], self.v_buffer[:, index]], dim=-2)}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        if not self._split_full_att_kv:
            return super().load_index_kv_buffer(index, load_tensor_dict)
        kv = load_tensor_dict["kv_buffer"]
        self.k_buffer[:, index].copy_(kv[..., : self.head_num, :])
        self.v_buffer[:, index].copy_(kv[..., self.head_num :, :])

    def _init_linear_att_buffers(self):
        big_page_token_num = (
            get_env_start_args().linear_att_page_block_num * get_env_start_args().linear_att_hash_page_size
        )
        # 申请大页可能需要对应的资源, 多申请了两个linear att的状态，理论上这个状态
        # 永远不会被 alloc 申请到，只会在 cpu cache中，用于过渡和存储碎页情况下的
        # cpu cache 的页面拷贝。
        self.linear_att_big_page_buffers = LinearAttCacheManager(
            size=triton.cdiv(self.size, big_page_token_num) + 2,
            linear_config=self.linear_config,
            keep_num=2,
        )

        self.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 2
        self.CPU_CACHE_BIG_PAGE_OFFLOAD_TEMP_BUFFER_ID = self.linear_att_big_page_buffers.size - 1
        return

    def _free_buffers(self):
        if self._split_full_att_kv:
            self.k_buffer = None
            self.v_buffer = None
            self.kv_buffer = None
        else:
            super()._free_buffers()
        self._free_linear_att_buffers()
        return

    def _free_linear_att_buffers(self):
        self.linear_att_big_page_buffers = None
        return

    def write_to_shm(self, req_manager):
        self.req_to_conv_state = req_manager.req_to_conv_state
        self.req_to_ssm_state = req_manager.req_to_ssm_state
        # super().write_to_shm() 会用 ForkingPickler 序列化本对象，torch 在 dump 时会把
        # CPU tensor 的 storage 原地迁到共享内存，使本进程大页 state cache 原本
        # pinned(cudaHostAlloc) 的内存退化为普通 shm mmap，之后 Triton kernel 携带该指针
        # 启动会报 "Pointer argument cannot be accessed from Triton (cpu tensor?)"。
        # 跨进程消费方并不使用 cpu 侧大页 state cache，序列化期间临时剔除以保住 pinned。
        big_page_buffers = self.linear_att_big_page_buffers
        self.linear_att_big_page_buffers = None
        try:
            return super().write_to_shm(req_manager)
        finally:
            self.linear_att_big_page_buffers = big_page_buffers

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        kv_move_buffer = super().alloc_paged_kv_move_buffer(page_num, page_size)
        Qwen3NextLinearAttPageHelper(self).assert_page_size()
        return kv_move_buffer

    def write_mem_to_page_kv_move_buffer(
        self,
        mem_indexes,
        page_index: int,
        dp_index: int,
        mem_managers,
        dp_world_size: int,
        page_kind: str = "kv",
        req_idx: int = None,
        target_tp_index: int = None,
    ):
        if page_kind == "kv":
            return super().write_mem_to_page_kv_move_buffer(
                mem_indexes=mem_indexes,
                page_index=page_index,
                dp_index=dp_index,
                mem_managers=mem_managers,
                dp_world_size=dp_world_size,
                page_kind=page_kind,
                req_idx=req_idx,
                target_tp_index=target_tp_index,
            )
        assert page_kind == "linear_att_state", f"unknown page_kind={page_kind}"
        assert req_idx is not None
        helper = Qwen3NextLinearAttPageHelper(self)
        dp_mems = helper.get_dp_mems(mem_managers, dp_index, dp_world_size)
        helper.write_req_to_page(
            page_index=page_index, req_idx=req_idx, dp_mems=dp_mems, target_tp_index=target_tp_index
        )
        return

    def read_page_kv_move_buffer_to_mem(
        self,
        mem_indexes,
        page_index: int,
        dp_index: int,
        mem_managers,
        dp_world_size: int,
        page_kind: str = "kv",
        req_idx: int = None,
        target_tp_index: int = None,
    ):
        if page_kind == "kv":
            return super().read_page_kv_move_buffer_to_mem(
                mem_indexes=mem_indexes,
                page_index=page_index,
                dp_index=dp_index,
                mem_managers=mem_managers,
                dp_world_size=dp_world_size,
                page_kind=page_kind,
                req_idx=req_idx,
                target_tp_index=target_tp_index,
            )
        assert page_kind == "linear_att_state", f"unknown page_kind={page_kind}"
        assert req_idx is not None
        helper = Qwen3NextLinearAttPageHelper(self)
        dp_mems = helper.get_dp_mems(mem_managers, dp_index, dp_world_size)
        helper.read_page_to_req(
            page_index=page_index, req_idx=req_idx, dp_mems=dp_mems, target_tp_index=target_tp_index
        )
        return


class Qwen3NextLinearAttPageHelper:
    def __init__(self, mem_manager: "Qwen3NextMemManager"):
        self.mem_manager = mem_manager
        self.linear_config = mem_manager.linear_config
        self.req_to_conv_state = mem_manager.req_to_conv_state
        self.req_to_ssm_state = mem_manager.req_to_ssm_state
        self.global_linear_k_heads = self.linear_config.global_linear_k_heads
        self.global_linear_v_heads = self.linear_config.global_linear_v_heads

        self.global_q_dim = self.global_linear_k_heads * self.linear_config.head_linear_k_dim
        self.global_k_dim = self.global_q_dim
        self.global_v_heads = self.global_linear_v_heads
        self.global_v_dim = self.global_v_heads * self.linear_config.head_linear_v_dim
        # conv state follows mixed_qkv layout: [q, k, v], each as a flat channel block.
        self.conv_shape = (
            self.linear_config.linear_layer_num,
            self.global_q_dim + self.global_k_dim + self.global_v_dim,
            self.linear_config.conv_kernel_size - 1,
        )
        self.ssm_shape = (
            self.linear_config.linear_layer_num,
            self.global_v_heads,
            self.linear_config.head_linear_k_dim,
            self.linear_config.head_linear_v_dim,
        )

        self.conv_nbytes = (
            self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2] * self.req_to_conv_state.buffer.element_size()
        )
        ssm_alignment = self.req_to_ssm_state.buffer.element_size()
        # 做一下字节对齐，防止切出来的不对齐，导致一些操作的性能下降。
        self.ssm_offset = ((self.conv_nbytes + ssm_alignment - 1) // ssm_alignment) * ssm_alignment
        self.ssm_nbytes = (
            self.ssm_shape[0]
            * self.ssm_shape[1]
            * self.ssm_shape[2]
            * self.ssm_shape[3]
            * self.req_to_ssm_state.buffer.element_size()
        )
        self.state_nbytes = self.ssm_offset + self.ssm_nbytes

    def assert_page_size(self):
        kv_move_buffer = self.mem_manager.kv_move_buffer
        page_nbytes = kv_move_buffer[0].numel() * kv_move_buffer.element_size()
        assert (
            page_nbytes >= self.state_nbytes
        ), f"nixl kv move page bytes {page_nbytes} is smaller than global linear att state bytes {self.state_nbytes}"
        return

    def get_dp_mems(self, mem_managers: List["Qwen3NextMemManager"], dp_index: int, dp_world_size: int):
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        assert len(dp_mems) == dp_world_size
        for mem in dp_mems:
            if mem is None:
                continue
            assert hasattr(mem, "req_to_conv_state") and hasattr(mem, "req_to_ssm_state")
            assert mem.linear_config.linear_layer_num == self.linear_config.linear_layer_num
            assert mem.linear_config.conv_kernel_size == self.linear_config.conv_kernel_size
            assert mem.linear_config.head_linear_k_dim == self.linear_config.head_linear_k_dim
            assert mem.linear_config.head_linear_v_dim == self.linear_config.head_linear_v_dim
            assert mem.linear_config.num_linear_k_heads == self.linear_config.num_linear_k_heads
            assert mem.linear_config.num_linear_v_heads == self.linear_config.num_linear_v_heads
        return dp_mems

    def view_page_to_linear_att_state(self, page_index: int):
        page_bytes = self.mem_manager.kv_move_buffer[page_index].view(torch.uint8).reshape(-1)
        conv_page = page_bytes[0 : self.conv_nbytes].view(self.req_to_conv_state.buffer.dtype).view(self.conv_shape)
        ssm_page = (
            page_bytes[self.ssm_offset : self.ssm_offset + self.ssm_nbytes]
            .view(self.req_to_ssm_state.buffer.dtype)
            .view(self.ssm_shape)
        )
        return conv_page, ssm_page

    def write_req_to_page(
        self,
        page_index: int,
        req_idx: int,
        dp_mems: List["Qwen3NextMemManager"],
        target_tp_index: int = None,
    ):
        conv_page, ssm_page = self.view_page_to_linear_att_state(page_index)
        for tp_index, mem in enumerate(dp_mems):
            if target_tp_index is not None and tp_index != target_tp_index:
                continue
            self._write_one_rank(mem, tp_index, req_idx, conv_page, ssm_page)
        return

    def read_page_to_req(
        self,
        page_index: int,
        req_idx: int,
        dp_mems: List["Qwen3NextMemManager"],
        target_tp_index: int = None,
    ):
        conv_page, ssm_page = self.view_page_to_linear_att_state(page_index)
        for tp_index, mem in enumerate(dp_mems):
            if target_tp_index is not None and tp_index != target_tp_index:
                continue
            self._read_one_rank(mem, tp_index, req_idx, conv_page, ssm_page)
        return

    def _select_conv_state(self, mem: "Qwen3NextMemManager", conv_req_idx: int) -> torch.Tensor:
        slot = mem.req_to_conv_state.buffer[:, conv_req_idx]
        window = self.conv_shape[-1]
        if mem.linear_config.conv_state_feat_last():
            return slot[:, :window, :].transpose(1, 2)
        return slot[..., :window]

    def _get_req_state_indexes(self, req_idx: int):
        mtp_size = get_env_start_args().mtp_step + 1
        # Conv is one widened slot per request; SSM keeps the historical S+1 block layout.
        return req_idx, req_idx * mtp_size

    def _write_one_rank(
        self,
        mem: "Qwen3NextMemManager",
        tp_index: int,
        req_idx: int,
        conv_page: torch.Tensor,
        ssm_page: torch.Tensor,
    ):
        conv_req_idx, ssm_req_idx = self._get_req_state_indexes(req_idx)
        conv_state = self._select_conv_state(mem, conv_req_idx)
        ssm_state = mem.req_to_ssm_state.buffer[:, ssm_req_idx, ...]
        self._copy_conv_state_to_page(conv_state, conv_page, mem, tp_index)
        self._copy_ssm_state_to_page(ssm_state, ssm_page, mem, tp_index)
        return

    def _copy_conv_state_to_page(
        self,
        conv_state: torch.Tensor,
        conv_page: torch.Tensor,
        mem: "Qwen3NextMemManager",
        tp_index: int,
    ):
        local_q_heads = mem.linear_config.num_linear_k_heads
        local_v_heads = mem.linear_config.num_linear_v_heads
        head_k_dim = mem.linear_config.head_linear_k_dim
        head_v_dim = mem.linear_config.head_linear_v_dim

        local_q_state = conv_state[:, 0 : local_q_heads * head_k_dim, :]
        local_k_state = conv_state[:, local_q_heads * head_k_dim : 2 * local_q_heads * head_k_dim, :]
        local_v_state = conv_state[:, 2 * local_q_heads * head_k_dim :, :]
        global_q_page = conv_page[:, 0 : self.global_q_dim, :]
        global_k_page = conv_page[:, self.global_q_dim : self.global_q_dim + self.global_k_dim, :]
        global_v_page = conv_page[:, self.global_q_dim + self.global_k_dim :, :]

        qk_head_slice = self._get_head_slice(
            tp_index, local_q_heads, self.global_linear_k_heads, mem.linear_config.tp_world_size, is_write=True
        )
        if qk_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = qk_head_slice
            local_dim_start = local_head_start * head_k_dim
            local_dim_end = local_head_end * head_k_dim
            global_dim_start = global_head_start * head_k_dim
            global_dim_end = global_head_end * head_k_dim
            global_q_page[:, global_dim_start:global_dim_end, :].copy_(
                local_q_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )
            global_k_page[:, global_dim_start:global_dim_end, :].copy_(
                local_k_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )

        v_head_slice = self._get_head_slice(
            tp_index, local_v_heads, self.global_linear_v_heads, mem.linear_config.tp_world_size, is_write=True
        )
        if v_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = v_head_slice
            local_dim_start = local_head_start * head_v_dim
            local_dim_end = local_head_end * head_v_dim
            global_dim_start = global_head_start * head_v_dim
            global_dim_end = global_head_end * head_v_dim
            global_v_page[:, global_dim_start:global_dim_end, :].copy_(
                local_v_state[:, local_dim_start:local_dim_end, :], non_blocking=True
            )
        return

    def _copy_ssm_state_to_page(
        self,
        ssm_state: torch.Tensor,
        ssm_page: torch.Tensor,
        mem: "Qwen3NextMemManager",
        tp_index: int,
    ):
        head_slice = self._get_head_slice(
            tp_index,
            mem.linear_config.num_linear_v_heads,
            self.global_linear_v_heads,
            mem.linear_config.tp_world_size,
            is_write=True,
        )
        if head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = head_slice
            ssm_page[:, global_head_start:global_head_end, :, :].copy_(
                ssm_state[:, local_head_start:local_head_end, :, :],
                non_blocking=True,
            )
        return

    def _get_head_slice(
        self,
        tp_index: int,
        local_heads: int,
        global_heads: int,
        tp_world_size: int,
        is_write: bool,
    ):
        if local_heads == 0 or global_heads == 0:
            return None
        total_local_heads = local_heads * tp_world_size
        repeat_count = max(1, total_local_heads // global_heads)
        if is_write and repeat_count > 1 and tp_index % repeat_count != 0:
            return None
        unique_tp_index = tp_index // repeat_count
        global_head_start = unique_tp_index * local_heads
        global_head_end = min(global_head_start + local_heads, global_heads)
        local_head_start = 0
        local_head_end = global_head_end - global_head_start
        if local_head_end <= local_head_start:
            return None
        return local_head_start, local_head_end, global_head_start, global_head_end

    def _copy_page_to_conv_state(
        self,
        conv_page: torch.Tensor,
        conv_state: torch.Tensor,
        mem: "Qwen3NextMemManager",
        tp_index: int,
    ):
        local_q_heads = mem.linear_config.num_linear_k_heads
        local_v_heads = mem.linear_config.num_linear_v_heads
        head_k_dim = mem.linear_config.head_linear_k_dim
        head_v_dim = mem.linear_config.head_linear_v_dim

        local_q_state = conv_state[:, 0 : local_q_heads * head_k_dim, :]
        local_k_state = conv_state[:, local_q_heads * head_k_dim : 2 * local_q_heads * head_k_dim, :]
        local_v_state = conv_state[:, 2 * local_q_heads * head_k_dim :, :]
        global_q_page = conv_page[:, 0 : self.global_q_dim, :]
        global_k_page = conv_page[:, self.global_q_dim : self.global_q_dim + self.global_k_dim, :]
        global_v_page = conv_page[:, self.global_q_dim + self.global_k_dim :, :]

        qk_head_slice = self._get_head_slice(
            tp_index, local_q_heads, self.global_linear_k_heads, mem.linear_config.tp_world_size, is_write=False
        )
        if qk_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = qk_head_slice
            local_dim_start = local_head_start * head_k_dim
            local_dim_end = local_head_end * head_k_dim
            global_dim_start = global_head_start * head_k_dim
            global_dim_end = global_head_end * head_k_dim
            local_q_state[:, local_dim_start:local_dim_end, :].copy_(
                global_q_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )
            local_k_state[:, local_dim_start:local_dim_end, :].copy_(
                global_k_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )

        v_head_slice = self._get_head_slice(
            tp_index, local_v_heads, self.global_linear_v_heads, mem.linear_config.tp_world_size, is_write=False
        )
        if v_head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = v_head_slice
            local_dim_start = local_head_start * head_v_dim
            local_dim_end = local_head_end * head_v_dim
            global_dim_start = global_head_start * head_v_dim
            global_dim_end = global_head_end * head_v_dim
            local_v_state[:, local_dim_start:local_dim_end, :].copy_(
                global_v_page[:, global_dim_start:global_dim_end, :], non_blocking=True
            )
        return

    def _copy_page_to_ssm_state(
        self,
        ssm_page: torch.Tensor,
        ssm_state: torch.Tensor,
        mem: "Qwen3NextMemManager",
        tp_index: int,
    ):
        head_slice = self._get_head_slice(
            tp_index,
            mem.linear_config.num_linear_v_heads,
            self.global_linear_v_heads,
            mem.linear_config.tp_world_size,
            is_write=False,
        )
        if head_slice is not None:
            local_head_start, local_head_end, global_head_start, global_head_end = head_slice
            ssm_state[:, local_head_start:local_head_end, :, :].copy_(
                ssm_page[:, global_head_start:global_head_end, :, :],
                non_blocking=True,
            )
        return

    def _read_one_rank(
        self,
        mem: "Qwen3NextMemManager",
        tp_index: int,
        req_idx: int,
        conv_page: torch.Tensor,
        ssm_page: torch.Tensor,
    ):
        conv_req_idx, ssm_req_idx = self._get_req_state_indexes(req_idx)
        conv_state = self._select_conv_state(mem, conv_req_idx)
        ssm_state = mem.req_to_ssm_state.buffer[:, ssm_req_idx, ...]
        self._copy_page_to_conv_state(conv_page, conv_state, mem, tp_index)
        self._copy_page_to_ssm_state(ssm_page, ssm_state, mem, tp_index)
        return

    def copy_rank_slice(self, dst_page: int, src_page: int, src_rank: int, src_world: int) -> None:
        # 把 prefill rank 写在全局页上的 linear head 挪到 decode 的累积页。其余 head 不动。
        k_local = self.global_linear_k_heads // src_world
        v_local = self.global_linear_v_heads // src_world
        k0 = src_rank * k_local
        v0 = src_rank * v_local
        head_k = self.linear_config.head_linear_k_dim
        head_v = self.linear_config.head_linear_v_dim
        src_conv, src_ssm = self.view_page_to_linear_att_state(src_page)
        dst_conv, dst_ssm = self.view_page_to_linear_att_state(dst_page)
        k_off = self.global_q_dim
        v_off = self.global_q_dim + self.global_k_dim
        q_slice = slice(k0 * head_k, (k0 + k_local) * head_k)
        k_slice = slice(k_off + k0 * head_k, k_off + (k0 + k_local) * head_k)
        v_slice = slice(v_off + v0 * head_v, v_off + (v0 + v_local) * head_v)
        dst_conv[:, q_slice, :].copy_(src_conv[:, q_slice, :], non_blocking=True)
        dst_conv[:, k_slice, :].copy_(src_conv[:, k_slice, :], non_blocking=True)
        dst_conv[:, v_slice, :].copy_(src_conv[:, v_slice, :], non_blocking=True)
        dst_ssm[:, v0 : v0 + v_local, :, :].copy_(src_ssm[:, v0 : v0 + v_local, :, :], non_blocking=True)

    def _local_linear_pieces(self, page_index: int, tp_index: int) -> List[torch.Tensor]:
        # 页上按全局 head 排布，本 rank 只写了自己那一段。把这几段按固定顺序收出来，NCCL 就不用送整页。
        conv_page, ssm_page = self.view_page_to_linear_att_state(page_index)

        pieces: List[torch.Tensor] = []
        qk_head_slice = self._get_head_slice(
            tp_index,
            self.linear_config.num_linear_k_heads,
            self.global_linear_k_heads,
            self.linear_config.tp_world_size,
            is_write=True,
        )
        if qk_head_slice is not None:
            _, _, global_head_start, global_head_end = qk_head_slice
            global_dim_start = global_head_start * self.linear_config.head_linear_k_dim
            global_dim_end = global_head_end * self.linear_config.head_linear_k_dim
            global_q_page = conv_page[:, 0 : self.global_q_dim, :]
            global_k_page = conv_page[:, self.global_q_dim : self.global_q_dim + self.global_k_dim, :]
            pieces.append(global_q_page[:, global_dim_start:global_dim_end, :])
            pieces.append(global_k_page[:, global_dim_start:global_dim_end, :])

        v_head_slice = self._get_head_slice(
            tp_index,
            self.linear_config.num_linear_v_heads,
            self.global_linear_v_heads,
            self.linear_config.tp_world_size,
            is_write=True,
        )
        if v_head_slice is not None:
            _, _, global_head_start, global_head_end = v_head_slice
            global_dim_start = global_head_start * self.linear_config.head_linear_v_dim
            global_dim_end = global_head_end * self.linear_config.head_linear_v_dim
            global_v_page = conv_page[:, self.global_q_dim + self.global_k_dim :, :]
            pieces.append(global_v_page[:, global_dim_start:global_dim_end, :])
            pieces.append(ssm_page[:, global_head_start:global_head_end, :, :])

        return pieces

    def _pack_buf_for(self, nbytes: int, device: torch.device) -> torch.Tensor:
        buf = getattr(self, "_pack_buf", None)
        if buf is None or buf.numel() < nbytes or buf.device != device:
            buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
            self._pack_buf = buf
        return buf

    def linear_pack_buffer(self, page_index: int, tp_index: int) -> torch.Tensor:
        cached = getattr(self, "_pack_nbytes", None)
        if cached is None or cached[0] != tp_index:
            pieces = self._local_linear_pieces(page_index, tp_index)
            nbytes = sum(piece.numel() * piece.element_size() for piece in pieces)
            self._pack_nbytes = (tp_index, nbytes)
        else:
            nbytes = cached[1]
        page = self.mem_manager.kv_move_buffer[page_index]
        return self._pack_buf_for(nbytes, page.device)[:nbytes]

    def _rank_linear_views(self, page_index: int, rank: int, world: int) -> List[torch.Tensor]:
        # 不等 TP 时按目的 rank 的 head 取段。2p4d 不用把 150MB 整页发给每张 decode 卡。
        if world <= 0 or self.global_linear_k_heads % world != 0 or self.global_linear_v_heads % world != 0:
            return []
        k_local = self.global_linear_k_heads // world
        v_local = self.global_linear_v_heads // world
        k0 = rank * k_local
        v0 = rank * v_local
        head_k = self.linear_config.head_linear_k_dim
        head_v = self.linear_config.head_linear_v_dim
        conv_page, ssm_page = self.view_page_to_linear_att_state(page_index)
        k_off = self.global_q_dim
        v_off = self.global_q_dim + self.global_k_dim
        return [
            conv_page[:, k0 * head_k : (k0 + k_local) * head_k, :],
            conv_page[:, k_off + k0 * head_k : k_off + (k0 + k_local) * head_k, :],
            conv_page[:, v_off + v0 * head_v : v_off + (v0 + v_local) * head_v, :],
            ssm_page[:, v0 : v0 + v_local, :, :],
        ]

    def rank_linear_buffer(self, page_index: int, rank: int, world: int, slot=None) -> torch.Tensor:
        # 4p2d 的一张 decode 卡同时收两路 prefill。缓冲只按 rank 留一份时，两路 stream 会互相盖掉。
        pieces = self._rank_linear_views(page_index, rank, world)
        nbytes = sum(piece.numel() * piece.element_size() for piece in pieces)
        page = self.mem_manager.kv_move_buffer[page_index]
        bufs = getattr(self, "_rank_recv_bufs", None)
        if bufs is None:
            bufs = {}
            self._rank_recv_bufs = bufs
        key = (slot, rank)
        buf = bufs.get(key)
        if buf is None or buf.numel() < nbytes or buf.device != page.device:
            buf = torch.empty(max(nbytes, 1), dtype=torch.uint8, device=page.device)
            bufs[key] = buf
        return buf[:nbytes]

    def pack_rank_linear(self, page_index: int, rank: int, world: int) -> torch.Tensor:
        pieces = self._rank_linear_views(page_index, rank, world)
        flats = [piece.contiguous().view(torch.uint8).reshape(-1) for piece in pieces]
        nbytes = sum(piece.numel() for piece in flats)
        page = self.mem_manager.kv_move_buffer[page_index]
        bufs = getattr(self, "_rank_pack_bufs", None)
        if bufs is None:
            bufs = {}
            self._rank_pack_bufs = bufs
        buf = bufs.get(rank)
        if buf is None or buf.numel() < nbytes or buf.device != page.device:
            buf = torch.empty(nbytes, dtype=torch.uint8, device=page.device)
            bufs[rank] = buf
        offset = 0
        for piece in flats:
            n = piece.numel()
            buf[offset : offset + n].copy_(piece)
            offset += n
        return buf[:nbytes]

    def unpack_rank_linear(self, page_index: int, rank: int, world: int, packed: torch.Tensor) -> None:
        pieces = self._rank_linear_views(page_index, rank, world)
        offset = 0
        for piece in pieces:
            nbytes = piece.numel() * piece.element_size()
            src = packed[offset : offset + nbytes].view(piece.dtype).reshape(piece.shape)
            piece.copy_(src)
            offset += nbytes

    def pack_local_linear(self, page_index: int, tp_index: int) -> torch.Tensor:
        pieces = self._local_linear_pieces(page_index, tp_index)
        flats = [piece.contiguous().view(torch.uint8).reshape(-1) for piece in pieces]
        nbytes = sum(piece.numel() for piece in flats)
        page = self.mem_manager.kv_move_buffer[page_index]
        buf = self._pack_buf_for(nbytes, page.device)
        offset = 0
        for piece in flats:
            n = piece.numel()
            buf[offset : offset + n].copy_(piece)
            offset += n
        return buf[:nbytes]

    def unpack_local_linear(self, page_index: int, tp_index: int, packed: torch.Tensor) -> None:
        pieces = self._local_linear_pieces(page_index, tp_index)
        offset = 0
        for piece in pieces:
            nbytes = piece.numel() * piece.element_size()
            src = packed[offset : offset + nbytes].view(piece.dtype).reshape(piece.shape)
            piece.copy_(src)
            offset += nbytes


def attach_maca_linear_pack(transporter, mem_manager) -> None:
    if not is_maca() or not hasattr(mem_manager, "linear_config"):
        return
    helper = Qwen3NextLinearAttPageHelper(mem_manager)
    transporter.maca_linear_pack = helper.pack_local_linear
    transporter.maca_linear_unpack = helper.unpack_local_linear
    transporter.maca_linear_buffer = helper.linear_pack_buffer
    transporter.maca_linear_pack_rank = helper.pack_rank_linear
    transporter.maca_linear_unpack_rank = helper.unpack_rank_linear
    transporter.maca_linear_rank_buffer = helper.rank_linear_buffer


class Qwen3NextInt8KVMemoryManager(Qwen3NextMemManager):

    operator_class = Int8KVLinearAttMemOperator
    npu_operator_class = NpuInt8KVLinearAttMemOperator

    def __init__(
        self,
        size: int | None,
        dtype: torch.dtype,
        num_kv_heads: int,
        head_dim: int,
        full_att_layer_num: int,
        linear_config: LinearAttCacheConfig,
        always_copy: bool = True,
        mem_fraction: float = 0.9,
    ) -> None:
        self.kv_dtype = torch.int8
        self.group_quant_size = get_env_start_args().llm_kv_quant_group_size
        # Must be set before super().__init__: profile_size() reads get_cell_size().
        self._split_full_att_kv = torch.device(get_target_device()).type == "npu"
        self.scale_dtype = torch.float32 if self._split_full_att_kv else dtype
        super().__init__(
            size,
            dtype,
            num_kv_heads,
            head_dim,
            full_att_layer_num,
            linear_config,
            always_copy=always_copy,
            mem_fraction=mem_fraction,
        )

    def get_cell_size(self) -> int:
        if self._split_full_att_kv:
            return NPUInt8KVMemoryManager.get_cell_size(self)
        kv_bytes = (
            2
            * self.head_num
            * self.head_dim
            * self.layer_num
            * torch._utils._element_size(self.kv_dtype)
        )
        scale_bytes = (
            2
            * self.head_num
            * (self.head_dim // self.group_quant_size)
            * self.layer_num
            * torch._utils._element_size(self.dtype)
        )
        return kv_bytes + scale_bytes

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        layer_index = self.get_full_att_cache_layer_index(layer_index)
        if self._split_full_att_kv:
            return NPUInt8KVMemoryManager.get_att_input_params(self, layer_index)
        k = self.kv_buffer[layer_index][:, : self.head_num, :]
        k_scale = self.scale_buffer[layer_index][:, : self.head_num, :]
        v = self.kv_buffer[layer_index][:, self.head_num :, :]
        v_scale = self.scale_buffer[layer_index][:, self.head_num :, :]
        return (k, k_scale), (v, v_scale)

    def get_prefill_att_input_params(
        self, kv: torch.Tensor, layer_index: int | None = None
    ) -> Tuple[Any, Any]:
        if layer_index is not None:
            layer_index = self.get_full_att_cache_layer_index(layer_index)
        if self._split_full_att_kv:
            return NPUInt8KVMemoryManager.get_prefill_att_input_params(self, kv, layer_index)

        k = kv[:, : self.head_num]
        v = kv[:, self.head_num :]
        if layer_index is None:
            return k, v
        return (
            k,
            self.kv_buffer[layer_index][:, : self.head_num, :],
            self.scale_buffer[layer_index][:, : self.head_num, :],
        ), (
            v,
            self.kv_buffer[layer_index][:, self.head_num :, :],
            self.scale_buffer[layer_index][:, self.head_num :, :],
        )

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        kv_move_buffer = super().alloc_paged_kv_move_buffer(page_num, page_size)
        scale_dim = self.scale_buffer.shape[-1]
        self.scale_move_buffer = torch.empty(
            (page_num, page_size, kv_move_buffer.shape[2], kv_move_buffer.shape[3], scale_dim),
            dtype=self.scale_dtype,
            device=self.target_device,
        )
        return kv_move_buffer

    def _init_buffers(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ) -> None:
        if self._split_full_att_kv:
            NPUInt8KVMemoryManager._init_buffers(self, size, dtype, head_num, head_dim, layer_num)
            self.kv_buffer = None
        else:
            self._init_group_int8_buffers(size, dtype, head_num, head_dim, layer_num)
        self._init_linear_att_buffers()

    def _init_group_int8_buffers(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ) -> None:
        group_size = self.group_quant_size
        if group_size <= 0 or (group_size & (group_size - 1)) != 0:
            raise ValueError(f"INT8 KV group size must be a power of 2, got {group_size}")
        if head_dim % group_size != 0:
            raise ValueError(
                f"INT8 KV cache requires head_dim to be a multiple of group size "
                f"{group_size}, got {head_dim}"
            )

        page_size = get_page_size()
        alloc_size = ((size // page_size) + 1) * page_size if page_size > 1 else size + 1
        kv_shape = (layer_num, alloc_size, 2 * head_num, head_dim)
        scale_shape = (layer_num, alloc_size, 2 * head_num, head_dim // group_size)
        logger.info(
            f"Total INT8 KV tokens allocated: {alloc_size} for page_size: {page_size}, "
            f"kv_shape: {kv_shape}, scale_shape: {scale_shape}, group_size: {group_size}"
        )
        self.kv_buffer = torch.empty(kv_shape, dtype=self.kv_dtype, device=self.target_device)
        self.scale_buffer = torch.empty(scale_shape, dtype=self.scale_dtype, device=self.target_device)
        hold_tokens = alloc_size - size
        self.kv_buffer[:, -hold_tokens:].zero_()
        self.scale_buffer[:, -hold_tokens:].fill_(1.0)

    def _free_buffers(self) -> None:
        if self._split_full_att_kv:
            NPUInt8KVMemoryManager._free_buffers(self)
        else:
            self.kv_buffer = None
            self.scale_buffer = None
        self._free_linear_att_buffers()

    def get_index_kv_buffer(self, index: Any) -> dict[str, torch.Tensor]:
        if self._split_full_att_kv:
            raise NotImplementedError("Qwen3.5 INT8 KV cache does not support prompt-cache export yet")
        return {
            "kv_buffer": self.kv_buffer[:, index],
            "scale_buffer": self.scale_buffer[:, index],
        }

    def load_index_kv_buffer(self, index: Any, load_tensor_dict: dict[str, torch.Tensor]) -> None:
        if self._split_full_att_kv:
            raise NotImplementedError("Qwen3.5 INT8 KV cache does not support prompt-cache import yet")
        self.kv_buffer[:, index].copy_(load_tensor_dict["kv_buffer"])
        self.scale_buffer[:, index].copy_(load_tensor_dict["scale_buffer"])


class CoreXQwen3NextMemManager(Qwen3NextMemManager):
    operator_class = CoreXLinearAttMemOperator
    _split_full_att_kv = False

    @staticmethod
    def supported(dtype: torch.dtype) -> bool:
        start_args = get_env_start_args()
        return (
            start_args.hardware_platform == "corex"
            and get_page_size() > 1
            and not start_args.enable_cpu_cache
            and dtype in (torch.float16, torch.bfloat16)
        )

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        layer_index = self.get_full_att_cache_layer_index(layer_index)
        return self.k_cache[layer_index], self.v_cache[layer_index]

    def _init_buffers(self, size: int, dtype: torch.dtype, head_num: int, head_dim: int, layer_num: int) -> None:
        page = get_page_size()
        alloc_size = ((size // page) + 1) * page
        num_blocks = alloc_size // page
        shape = (layer_num, num_blocks, head_num, page, head_dim)
        self.k_cache = torch.empty(shape, dtype=dtype, device=self.target_device)
        self.v_cache = torch.empty(shape, dtype=dtype, device=self.target_device)

        self.kv_buffer = self.k_cache
        logger.info(f"CoreX paged KV cache {tuple(shape)} page {page}")
        from lightllm.common.basemodel.attention.ixformer_corex import CorexBlockTable

        self.block_table = CorexBlockTable(self.target_device)
        logger.info(
            f"CoreX block table rows {self.block_table.rows} pages {self.block_table.pages}"
        )
        self._init_linear_att_buffers()

    def _token_index(self, index) -> torch.Tensor:
        if not torch.is_tensor(index):
            index = torch.tensor(index, dtype=torch.int64, device=self.k_cache.device)
        return index.to(device=self.k_cache.device, dtype=torch.int64)

    def _page_slot(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        token = self._token_index(index)
        page = self.k_cache.shape[3]
        block = torch.div(token, page, rounding_mode="floor")
        offset = token.remainder(page)
        return block, offset

    def _gather_heads(self, cache: torch.Tensor, index) -> torch.Tensor:
        block, offset = self._page_slot(index)
        picked = cache[:, block, :, offset, :]
        if block.ndim == 0:
            return picked
        return picked.movedim(0, 1)

    def _scatter_heads(self, cache: torch.Tensor, index, values: torch.Tensor) -> None:
        block, offset = self._page_slot(index)
        if block.ndim != 0:
            values = values.movedim(0, 1)
        cache[:, block, :, offset, :] = values

    def _packed_token_kv(self, index) -> torch.Tensor:
        k = self._gather_heads(self.k_cache, index)
        v = self._gather_heads(self.v_cache, index)
        return torch.cat((k, v), dim=-2)

    def get_index_kv_buffer(self, index: Any) -> dict[str, torch.Tensor]:
        return {"kv_buffer": self._packed_token_kv(index)}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        kv = load_tensor_dict["kv_buffer"]
        self._scatter_heads(self.k_cache, index, kv[..., : self.head_num, :])
        self._scatter_heads(self.v_cache, index, kv[..., self.head_num :, :])

    def _free_buffers(self):
        self.k_cache = None
        self.v_cache = None
        self.kv_buffer = None
        self.block_table = None
        self._free_linear_att_buffers()
