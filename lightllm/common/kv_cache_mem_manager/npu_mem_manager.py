from lightllm.common.kv_cache_mem_manager.operator.base import BaseMemManagerOperator
import torch
from typing import Any, List, Tuple

from lightllm.utils.envs_utils import get_page_size
from lightllm.utils.log_utils import init_logger

from .mem_manager import MemoryManager


logger = init_logger(__name__)


NPU_INT8_KV_ALIGNMENT = 32


class NPUOperator(BaseMemManagerOperator):

    def __init__(self, mem_manager: "MemoryManager") -> None:
        super().__init__(mem_manager)
        self._paged_kv_views = None

    def _get_paged_kv_views(self):
        if self._paged_kv_views is not None:
            return self._paged_kv_views

        page_size = get_page_size()
        kb0 = self.mem_manager.k_buffer[0]
        if not kb0.is_contiguous() or not self.mem_manager.v_buffer[0].is_contiguous():
            self._paged_kv_views = []
            return self._paged_kv_views

        num_blocks = kb0.shape[0] // page_size
        n_kv, head_dim = kb0.shape[1], kb0.shape[2]
        self._paged_kv_views = [
            (
                self.mem_manager.k_buffer[i].view(num_blocks, page_size, n_kv, head_dim),
                self.mem_manager.v_buffer[i].view(num_blocks, page_size, n_kv, head_dim),
            )
            for i in range(self.mem_manager.k_buffer.shape[0])
        ]
        return self._paged_kv_views

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        kb, vb = self.mem_manager.k_buffer[layer_index], self.mem_manager.v_buffer[layer_index]
        k_src, v_src = kv[:, : self.mem_manager.head_num, :], kv[:, self.mem_manager.head_num :, :]
        assert kv.shape[0] == mem_index.shape[0], (kv.shape, mem_index.shape)
        assert k_src.shape[1] == kb.shape[1] and k_src.shape[2] == kb.shape[2], (k_src.shape, kb.shape)
        assert v_src.shape[1] == vb.shape[1] and v_src.shape[2] == vb.shape[2], (v_src.shape, vb.shape)

        views = self._get_paged_kv_views()
        if views:
            import torch_npu

            key_cache, value_cache = views[layer_index]
            slot = mem_index if mem_index.dtype == torch.int32 else mem_index.to(torch.int32)
            # _npu_reshape_and_cache requires layout: [num_blocks, page_size, n_kv, head_dim]
            torch_npu._npu_reshape_and_cache(
                key=k_src,
                value=v_src,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_indices=slot,
            )
            return

        kb.index_copy_(0, mem_index, k_src)
        vb.index_copy_(0, mem_index, v_src)


class NPUMemoryManager(MemoryManager):
    operator_class = NPUOperator

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        return self.k_buffer[layer_index], self.v_buffer[layer_index]

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        page_size = get_page_size()
        alloc_size = ((size // page_size) + 1) * page_size if page_size > 1 else size + 1
        logger.info(f"Total page blocks allocated: {alloc_size // page_size} for page_size: {page_size}")
        self.k_buffer = torch.empty((layer_num, alloc_size, head_num, head_dim), dtype=dtype, device=self.target_device)
        self.v_buffer = torch.empty((layer_num, alloc_size, head_num, head_dim), dtype=dtype, device=self.target_device)
        self.kv_buffer = self.k_buffer

    def _free_buffers(self):
        self.k_buffer = None
        self.v_buffer = None
        self.kv_buffer = None

    def get_index_kv_buffer(self, index):
        return {
            "kv_buffer": torch.cat([self.k_buffer[:, index], self.v_buffer[:, index]], dim=1),
        }

    def load_index_kv_buffer(self, index, load_tensor_dict):
        t = load_tensor_dict["kv_buffer"]
        self.k_buffer[:, index].copy_(t[:, : self.head_num])
        self.v_buffer[:, index].copy_(t[:, self.head_num :])

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        raise NotImplementedError("NPUMemoryManager does not support PD-separated alloc_paged_kv_move_buffer")

    def write_mem_to_page_kv_move_buffer(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
        page_kind: str = "kv",
        req_idx: int = None,
    ):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated write_mem_to_page_kv_move_buffer")

    def read_page_kv_move_buffer_to_mem(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
        page_kind: str = "kv",
        req_idx: int = None,
    ):
        raise NotImplementedError("NPUMemoryManager does not support PD-separated read_page_kv_move_buffer_to_mem")


class NPUInt8KVOperator(BaseMemManagerOperator):

    def copy_kv_to_mem_manager(
        self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor
    ) -> None:
        import torch_npu

        mem_manager = self.mem_manager
        token_num = kv.shape[0]
        if token_num != mem_index.shape[0]:
            raise ValueError(f"KV token count {token_num} does not match index count {mem_index.shape[0]}")
        if token_num == 0:
            return

        expected_shape = (token_num, 2 * mem_manager.head_num, mem_manager.head_dim)
        if tuple(kv.shape) != expected_shape:
            raise ValueError(f"Expected KV shape {expected_shape}, got {tuple(kv.shape)}")

        k_src = kv[:, : mem_manager.head_num].reshape(token_num, -1)
        v_src = kv[:, mem_manager.head_num :].reshape(token_num, -1)
        k_quant, k_scale = torch_npu.npu_dynamic_quant(k_src, dst_type=torch.int8)
        v_quant, v_scale = torch_npu.npu_dynamic_quant(v_src, dst_type=torch.int8)

        slot = mem_index.contiguous()
        if slot.dtype not in (torch.int32, torch.int64):
            slot = slot.to(torch.int32)

        torch_npu._npu_reshape_and_cache(
            key=k_quant.view(token_num, mem_manager.head_num, mem_manager.head_dim),
            value=v_quant.view(token_num, mem_manager.head_num, mem_manager.head_dim),
            key_cache=mem_manager.k_buffer[layer_index],
            value_cache=mem_manager.v_buffer[layer_index],
            slot_indices=slot,
        )

        mem_manager.k_scale_buffer[layer_index].view(-1).index_copy_(
            0,
            slot,
            k_scale,
        )
        mem_manager.v_scale_buffer[layer_index].view(-1).index_copy_(
            0,
            slot,
            v_scale,
        )


class NPUInt8KVMemoryManager(NPUMemoryManager):

    operator_class = NPUInt8KVOperator

    def __init__(
        self,
        size: int | None,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        always_copy: bool = True,
        mem_fraction: float = 0.9,
    ) -> None:
        self.kv_dtype = torch.int8
        self.scale_dtype = torch.float32
        super().__init__(
            size,
            dtype,
            head_num,
            head_dim,
            layer_num,
            always_copy=always_copy,
            mem_fraction=mem_fraction,
        )

    def get_cell_size(self) -> int:
        kv_bytes = 2 * self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.kv_dtype)
        scale_bytes = 2 * self.layer_num * torch._utils._element_size(self.scale_dtype)
        return kv_bytes + scale_bytes

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        return (
            self.k_buffer[layer_index],
            self.k_scale_buffer[layer_index],
        ), (
            self.v_buffer[layer_index],
            self.v_scale_buffer[layer_index],
        )

    def get_prefill_att_input_params(
        self, kv: torch.Tensor, layer_index: int | None = None
    ) -> Tuple[Any, Any]:
        k = kv[:, : self.head_num]
        v = kv[:, self.head_num :]
        if layer_index is None:
            return k, v

        return (
            k,
            self.k_buffer[layer_index],
            self.k_scale_buffer[layer_index],
        ), (
            v,
            self.v_buffer[layer_index],
            self.v_scale_buffer[layer_index],
        )

    def _init_buffers(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ) -> None:
        page_size = get_page_size()
        if page_size <= 1 or page_size > 512 or page_size % NPU_INT8_KV_ALIGNMENT != 0:
            raise ValueError(
                "Ascend INT8 paged KV cache requires PAGE_SIZE to be a multiple of "
                f"{NPU_INT8_KV_ALIGNMENT} and no greater than 512, got {page_size}"
            )

        if head_dim % NPU_INT8_KV_ALIGNMENT != 0:
            raise ValueError(
                f"Ascend INT8 paged KV cache requires head_dim to be a multiple of "
                f"{NPU_INT8_KV_ALIGNMENT}, got {head_dim}"
            )

        alloc_size = ((size // page_size) + 1) * page_size
        num_blocks = alloc_size // page_size
        cache_shape = (layer_num, num_blocks, page_size, head_num, head_dim)
        scale_shape = (layer_num, num_blocks, page_size)
        logger.info(
            f"Total INT8 KV page blocks allocated: {num_blocks} for page_size: {page_size}, "
            f"cache_shape: {cache_shape}, scale_shape: {scale_shape}"
        )

        self.k_buffer = torch.empty(cache_shape, dtype=self.kv_dtype, device=self.target_device)
        self.v_buffer = torch.empty(cache_shape, dtype=self.kv_dtype, device=self.target_device)
        self.k_scale_buffer = torch.empty(scale_shape, dtype=self.scale_dtype, device=self.target_device)
        self.v_scale_buffer = torch.empty(scale_shape, dtype=self.scale_dtype, device=self.target_device)
        # Decode graph warmup maps its dummy sequence to this reserved page.
        self.k_buffer[:, -1].zero_()
        self.v_buffer[:, -1].zero_()
        self.k_scale_buffer[:, -1].fill_(1.0)
        self.v_scale_buffer[:, -1].fill_(1.0)
        # Some common code uses kv_buffer only to identify the owning layer.
        self.kv_buffer = self.k_buffer

    def _free_buffers(self) -> None:
        self.k_buffer = None
        self.v_buffer = None
        self.k_scale_buffer = None
        self.v_scale_buffer = None
        self.kv_buffer = None

    def get_index_kv_buffer(self, index: Any) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Ascend INT8 KV cache does not support prompt-cache export yet")

    def load_index_kv_buffer(
        self, index: Any, load_tensor_dict: dict[str, torch.Tensor]
    ) -> None:
        raise NotImplementedError("Ascend INT8 KV cache does not support prompt-cache import yet")
