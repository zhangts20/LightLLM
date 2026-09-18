import dataclasses
import os
from typing import TYPE_CHECKING, Any, Callable

import torch

from lightllm.platform.base.attention import register_att_backend
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from ..base_att import AttControl
from .fp import (
    PagedFa3AttBackend,
    PagedFa3DecodeAttState,
    PagedFa3PrefillAttState,
    maca_flash_attn_varlen_func,
)
from .prefix_flash_npu import can_use_prefix_flash, prefix_flash_attention
from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import gather_dequant_int8kv

try:
    from flash_attn import flash_attn_func as maca_flash_attn_func
except ImportError:
    maca_flash_attn_func = None

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


NPU_PAGED_PER_TOKEN_ANTIQUANT_MODE = 4
DEFAULT_DEQUANT_CHUNK_TOKENS = 65536
DEFAULT_DEQUANT_RESERVE_MIB = 512
DEFAULT_MACA_DEQUANT_CHUNK_TOKENS = 16384
DEFAULT_MACA_ONESHOT_MAX_BYTES = 2 * 1024 ** 3
logger = init_logger(__name__)


def _merge_attn_lse(
    acc_out: torch.Tensor,
    acc_lse: torch.Tensor,
    part_out: torch.Tensor,
    part_lse: torch.Tensor,
    weighted_part: torch.Tensor,
    initialized: bool,
) -> None:
    if not initialized:
        acc_out.copy_(part_out)
        acc_lse.copy_(part_lse)
        return

    merged_lse = torch.logaddexp(acc_lse, part_lse)
    acc_out.mul_(torch.exp(acc_lse - merged_lse))
    torch.mul(part_out, torch.exp(part_lse - merged_lse), out=weighted_part)
    acc_out.add_(weighted_part)
    acc_lse.copy_(merged_lse)


@register_att_backend(
    name="paged_fa3",
    category="standard",
    kv_types=("int8kv",),
    platforms=("ascend",),
    validate_name="fa3",
)
class PagedFa3Int8KVAscendAttBackend(PagedFa3AttBackend):

    def create_att_prefill_state(
        self, infer_state: "InferStateInfo"
    ) -> "PagedFa3Int8KVAscendPrefillAttState":
        return PagedFa3Int8KVAscendPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(
        self, infer_state: "InferStateInfo"
    ) -> "PagedFa3Int8KVAscendDecodeAttState":
        return PagedFa3Int8KVAscendDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class PagedFa3Int8KVAscendPrefillAttState(PagedFa3PrefillAttState):
    # The request info for each prefill request, as a tuple of (q_start, q_end, req_id, prefix_len).
    # Then use req_id to index stored K/V cache and prefix_len to determine how many tokens to 
    # dequantize in chunks.
    request_slices: tuple = None
    # Use `npu_fused_infer_attention_score` when generating a single token and KV dequantization is required.
    use_paged_int8: bool = False
    # The prefill implementation to use. Can be "triton" or "dequant_fia". Default is "dequant_fia".
    prefill_impl: str = "dequant_fia"
    # Use in `dequant_fia`, the chunk size of tokens to dequantize at a time. Default is 65536. 
    dequant_chunk_tokens: int = DEFAULT_DEQUANT_CHUNK_TOKENS
    # The reserved memory size for dequantization, in bytes. Default is 512 MiB.
    dequant_reserve_bytes: int = DEFAULT_DEQUANT_RESERVE_MIB * 2**20
    # The number of tokens selected for dequantization in the current prefill state. 
    selected_dequant_chunk_tokens: int = 0
    # The key to identify the dequantization chunk configuration. If the current configuration matches 
    # this key, reuse the previous selected chunk size.
    dequant_chunk_key: tuple = None
    # The maximum prefix length among all requests in the current prefill state.
    max_prefix_len: int = 0

    def init_state(self) -> None:
        # ModelInput now carries total KV rows and current query rows.
        # Their difference is the cached prefix count, without a device read.
        self.prefix_total_token_num = self.infer_state.total_token_num - self.infer_state.input_ids.shape[0]
        self.request_slices = None
        self.use_paged_int8 = False
        self.selected_dequant_chunk_tokens = 0
        self.dequant_chunk_key = None
        self.max_prefix_len = 0

        self.prefill_impl = os.getenv(
            "LIGHTLLM_ASCEND_INT8KV_PREFILL_IMPL", "dequant_fia"
        ).lower()
        if self.prefill_impl not in ("triton", "dequant_fia"):
            raise ValueError(
                "LIGHTLLM_ASCEND_INT8KV_PREFILL_IMPL must be 'triton' or 'dequant_fia', "
                f"got {self.prefill_impl!r}"
            )

        self.dequant_chunk_tokens = int(
            os.getenv(
                "LIGHTLLM_ASCEND_INT8KV_DEQUANT_CHUNK_TOKENS",
                str(DEFAULT_DEQUANT_CHUNK_TOKENS),
            )
        )

        if self.dequant_chunk_tokens <= 0:
            raise ValueError(
                "LIGHTLLM_ASCEND_INT8KV_DEQUANT_CHUNK_TOKENS must be greater than zero"
            )

        reserve_mib = int(
            os.getenv(
                "LIGHTLLM_ASCEND_INT8KV_DEQUANT_RESERVE_MIB",
                str(DEFAULT_DEQUANT_RESERVE_MIB),
            )
        )
        if reserve_mib < 0:
            raise ValueError(
                "LIGHTLLM_ASCEND_INT8KV_DEQUANT_RESERVE_MIB must not be negative"
            )

        self.dequant_reserve_bytes = reserve_mib * 2**20
        if (
            self.prefix_total_token_num != 0
            and self.infer_state.max_q_seq_len == 1
        ):
            PagedFa3PrefillAttState.init_state(self)
            self.use_paged_int8 = True
            return

        self.atten_mask = self.backend.get_causal_attn_mask(
            self.infer_state.input_ids.device
        )

        if self.prefix_total_token_num == 0:
            return

        if self.prefill_impl == "triton":
            return

        # Generate request slices for each prefill request, which will be used to determine the 
        # K/V cache and prefix length for dequantization.
        q_ends = self.infer_state.b1_cu_q_seq_len_cpu.tolist()
        kv_lens = self.infer_state.b_cu_kv_seq_len_cpu.tolist()
        req_ids = self.infer_state.b_req_idx.detach().cpu().tolist()
        q_start = 0
        max_prefix_len = 0
        slices = []
        for q_end, kv_len, req_id in zip(q_ends, kv_lens, req_ids):
            q_len = q_end - q_start
            prefix_len = kv_len - q_len
            if q_len <= 0 or prefix_len < 0:
                raise ValueError(
                    f"Invalid prefill lengths: q_len={q_len}, kv_len={kv_len}"
                )
            max_prefix_len = max(max_prefix_len, prefix_len)
            slices.append((q_start, q_end, int(req_id), prefix_len))
            q_start = q_end
        self.request_slices = tuple(slices)
        self.max_prefix_len = max_prefix_len

    def _select_dequant_chunk_tokens(
        self, q: torch.Tensor, n_kv: int, head_dim: int
    ) -> int:
        if self.max_prefix_len <= 0:
            return 0

        # If the key matches the previous one, reuse the previously selected chunk size.
        key = (
            q.device,
            q.dtype,
            q.shape[-2],
            n_kv,
            head_dim,
            self.max_prefix_len,
            self.dequant_chunk_tokens,
            self.dequant_reserve_bytes,
        )
        if self.dequant_chunk_key == key:
            return self.selected_dequant_chunk_tokens

        page_size = self.backend.page_size
        configured_target_tokens = min(self.dequant_chunk_tokens, self.max_prefix_len)

        # The memory required for dequantization of each token, including the scratch space for quantized K/V, 
        scratch_bytes_per_token = (
            n_kv * head_dim * (1 + 2 * q.element_size()) + torch.float32.itemsize
        )

        merge_bytes = (
            self.infer_state.max_q_seq_len
            * q.shape[-2]
            * head_dim
            * (4 + q.element_size() + 4)
        )
        reserve_bytes = max(self.dequant_reserve_bytes, merge_bytes + 256 * 2**20)

        def fit_chunk_tokens(free_memory_bytes: int) -> int:
            available_scratch_bytes = max(0, free_memory_bytes - reserve_bytes)
            memory_limited_tokens = available_scratch_bytes // scratch_bytes_per_token
            fitted_tokens = min(configured_target_tokens, memory_limited_tokens)
            if fitted_tokens >= page_size:
                fitted_tokens = fitted_tokens // page_size * page_size
            return int(fitted_tokens)

        free_bytes, total_bytes = torch.npu.mem_get_info(q.device.index)
        initial_free_bytes = free_bytes
        target_tokens = fit_chunk_tokens(free_bytes)

        if target_tokens <= 0:
            allocated_bytes = torch.npu.memory_allocated(q.device.index)
            reserved_bytes = torch.npu.memory_reserved(q.device.index)
            reserved_minus_allocated_bytes = max(0, reserved_bytes - allocated_bytes)
            if reserved_minus_allocated_bytes > 0:
                torch.npu.empty_cache()
                free_bytes, total_bytes = torch.npu.mem_get_info(q.device.index)
                target_tokens = fit_chunk_tokens(free_bytes)
                if target_tokens > 0:
                    logger.warning(
                        "Recovered unused NPU allocator cache for INT8 KV "
                        "dequant: free_before=%.1f MiB, free_after=%.1f MiB, "
                        "reclaimable_cache=%.1f MiB, selected_chunk_tokens=%d",
                        initial_free_bytes / 2**20,
                        free_bytes / 2**20,
                        reserved_minus_allocated_bytes / 2**20,
                        target_tokens,
                    )

        if target_tokens <= 0:
            minimum_scratch_mib = scratch_bytes_per_token / 2**20
            raise RuntimeError(
                "Not enough free NPU memory for one INT8 KV dequant chunk: "
                f"free_before_cleanup={initial_free_bytes / 2**20:.1f} MiB, "
                f"free_after_cleanup={free_bytes / 2**20:.1f} MiB, "
                f"total={total_bytes / 2**20:.1f} MiB, "
                "reclaimable_cache_before_cleanup="
                f"{reserved_minus_allocated_bytes / 2**20:.1f} MiB, "
                f"safety_reserve={reserve_bytes / 2**20:.1f} MiB, "
                f"minimum_scratch={minimum_scratch_mib:.3f} MiB, "
                f"configured_chunk_tokens={self.dequant_chunk_tokens}, "
                f"batch_size={self.infer_state.batch_size}, "
                f"max_q_seq_len={self.infer_state.max_q_seq_len}, "
                f"max_prefix_len={self.max_prefix_len}, n_q={q.shape[-2]}, "
                f"n_kv={n_kv}, head_dim={head_dim}, page_size={page_size}. "
                "Recovery: stop clients, restart the API server and all TP ranks after old "
                "rank processes release NPU memory, then reduce --mem_fraction, "
                "--max_total_token_num, or --chunked_prefill_size as applicable. Do not lower "
                "LIGHTLLM_ASCEND_INT8KV_DEQUANT_RESERVE_MIB merely to suppress this error, "
                "because that can turn it into an actual NPU OOM."
            )
        self.selected_dequant_chunk_tokens = int(target_tokens)
        self.dequant_chunk_key = key
        return self.selected_dequant_chunk_tokens

    @staticmethod
    def _merge_fia_part(
        acc_out: torch.Tensor,
        acc_lse: torch.Tensor,
        part_out: torch.Tensor,
        part_lse: torch.Tensor,
        weighted_part: torch.Tensor,
        initialized: bool,
    ) -> None:
        if not initialized:
            acc_out.copy_(part_out)
            acc_lse.copy_(part_lse)
            return

        merged_lse = torch.logaddexp(acc_lse, part_lse)
        acc_out.mul_(torch.exp(acc_lse - merged_lse))
        torch.mul(
            part_out,
            torch.exp(part_lse - merged_lse),
            out=weighted_part,
        )
        acc_out.add_(weighted_part)
        acc_lse.copy_(merged_lse)

    @staticmethod
    def _normalize_bsnd_fia_outputs(
        part_out: torch.Tensor, part_lse: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return part_out.squeeze(0), part_lse.transpose(1, 2).squeeze(0)

    def _normal_prefill_att(
        self,
        q: torch.Tensor,
        k: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        v: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        att_control: AttControl,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        import torch_npu

        N_Q, HEAD_DIM = q.shape[-2:]
        if att_control.use_att_sink:
            raise NotImplementedError(
                "Ascend INT8 KV prefill does not support attention sink"
            )

        if self.use_paged_int8:
            if att_control.use_sliding_window:
                raise NotImplementedError(
                    "Ascend paged INT8 KV prefill does not support sliding-window attention"
                )
            _, k, k_scale = k
            _, v, v_scale = v
            N_KV = k.shape[-2]
            k = k.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
            v = v.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
            return torch_npu.npu_fused_infer_attention_score(
                query=q.unsqueeze(2),
                key=k,
                value=v,
                input_layout="BNSD",
                sparse_mode=0,
                scale=HEAD_DIM**-0.5,
                actual_seq_lengths_kv=self.infer_state.b_cu_kv_seq_len_cpu,
                num_heads=N_Q,
                num_key_value_heads=N_KV,
                block_table=self.page_table,
                block_size=self.backend.page_size,
                key_antiquant_scale=k_scale,
                value_antiquant_scale=v_scale,
                key_antiquant_mode=NPU_PAGED_PER_TOKEN_ANTIQUANT_MODE,
                value_antiquant_mode=NPU_PAGED_PER_TOKEN_ANTIQUANT_MODE,
            )[0].squeeze(2)

        # 1. No dequantization is needed, use the standard fused attention kernel.
        # 2. use_paged_int8 is True (above).
        # 3. The prefill implementation is "triton".
        if self.request_slices is None:
            fresh_k, k_cache, k_scale_cache = k
            fresh_v, v_cache, v_scale_cache = v
            if (
                self.prefix_total_token_num != 0
                and self.prefill_impl == "triton"
            ):
                from lightllm.common.basemodel.triton_kernel.att.prefill_att.context_flashattention_int8kv import (
                    context_attention_fwd_int8kv,
                )

                if att_control.use_sliding_window:
                    sliding_window = att_control.sliding_window
                else:
                    sliding_window = (-1, -1)
                output = alloc_func(q.shape, dtype=q.dtype, device=q.device)
                return context_attention_fwd_int8kv(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    k_scale=k_scale_cache,
                    v_scale=v_scale_cache,
                    out=output,
                    b_req_idx=self.infer_state.b_req_idx,
                    b_start_loc=self.infer_state.b_q_start_loc,
                    b_seq_len=self.infer_state.b_seq_len,
                    b_prompt_cache_len=self.infer_state.b_ready_cache_len,
                    max_q_len=self.infer_state.max_q_seq_len,
                    req_to_token_indexs=self.infer_state.req_manager.req_to_token_indexs,
                    sliding_window=sliding_window,
                )

            k = fresh_k
            v = fresh_v
            N_KV = k.shape[-2]
            actual_seq_lengths_kv = self.infer_state.b1_cu_q_seq_len_cpu
        else:
            if att_control.use_sliding_window:
                raise NotImplementedError(
                    "Chunked Ascend INT8 KV dequant prefill does not support sliding-window attention"
                )

            fresh_k, k_cache, k_scale_cache = k
            fresh_v, v_cache, v_scale_cache = v
            N_KV = k_cache.shape[-2]
            k_cache = k_cache.flatten(0, 1)
            v_cache = v_cache.flatten(0, 1)
            k_scale_cache = k_scale_cache.flatten()
            v_scale_cache = v_scale_cache.flatten()
            output = alloc_func(q.shape, dtype=q.dtype, device=q.device)
            chunk_tokens = self._select_dequant_chunk_tokens(q, N_KV, HEAD_DIM)
            # The temp workspace for dequantization of each chunk.
            quant_scratch = alloc_func(
                (chunk_tokens, N_KV, HEAD_DIM), dtype=torch.int8, device=q.device
            )
            scale_scratch = alloc_func(
                (chunk_tokens,), dtype=torch.float32, device=q.device
            )
            k_scratch = alloc_func(
                (chunk_tokens, N_KV, HEAD_DIM), dtype=q.dtype, device=q.device
            )
            v_scratch = alloc_func(
                (chunk_tokens, N_KV, HEAD_DIM), dtype=q.dtype, device=q.device
            )
            req_to_token = self.infer_state.req_manager.req_to_token_indexs

            # Dequantize the K/V cache in chunks and compute attention for each request slice.
            for q_start, q_end, req_id, prefix_len in self.request_slices:
                q_part = q[q_start:q_end]
                q_len = q_end - q_start

                # Allocate temporary buffers for the fused attention kernel outputs and intermediate accumulations.
                fia_out = alloc_func(
                    (1, q_len, N_Q, HEAD_DIM), dtype=q.dtype, device=q.device
                )
                fia_lse = alloc_func(
                    (1, N_Q, q_len, 1), dtype=torch.float32, device=q.device
                )
                acc_out = alloc_func(q_part.shape, dtype=torch.float32, device=q.device)
                acc_lse = alloc_func(
                    (q_len, N_Q, 1), dtype=torch.float32, device=q.device
                )

                weighted_part = alloc_func(
                    q_part.shape, dtype=torch.float32, device=q.device
                )
                acc_initialized = False

                # Dequantize the K/V cache in chunks and compute attention for each chunk.
                for chunk_start in range(0, prefix_len, chunk_tokens):
                    chunk_end = min(prefix_len, chunk_start + chunk_tokens)
                    token_count = chunk_end - chunk_start
                    indices = req_to_token[req_id, chunk_start:chunk_end]
                    quant_view = quant_scratch[:token_count]
                    scale_view = scale_scratch[:token_count]
                    k_view = k_scratch[:token_count]
                    v_view = v_scratch[:token_count]

                    torch.index_select(k_cache, 0, indices, out=quant_view)
                    torch.index_select(k_scale_cache, 0, indices, out=scale_view)
                    torch.mul(quant_view, scale_view.view(-1, 1, 1), out=k_view)
                    torch.index_select(v_cache, 0, indices, out=quant_view)
                    torch.index_select(v_scale_cache, 0, indices, out=scale_view)
                    torch.mul(quant_view, scale_view.view(-1, 1, 1), out=v_view)

                    if can_use_prefix_flash(q_part, k_view):
                        part_out, part_lse = prefix_flash_attention(q_part, k_view, v_view)
                    else:
                        torch_npu.npu_fused_infer_attention_score.out(
                            query=q_part.unsqueeze(0),
                            key=k_view.unsqueeze(0),
                            value=v_view.unsqueeze(0),
                            input_layout="BSND",
                            sparse_mode=0,
                            scale=HEAD_DIM**-0.5,
                            actual_seq_lengths=[q_len],
                            actual_seq_lengths_kv=[token_count],
                            num_heads=N_Q,
                            num_key_value_heads=N_KV,
                            softmax_lse_flag=True,
                            out=[fia_out, fia_lse],
                        )
                        # Merge the outputs of the current chunk into the accumulated outputs.
                        part_out, part_lse = self._normalize_bsnd_fia_outputs(
                            fia_out, fia_lse
                        )
                    self._merge_fia_part(
                        acc_out,
                        acc_lse,
                        part_out,
                        part_lse,
                        weighted_part,
                        acc_initialized,
                    )
                    acc_initialized = True

                # The attention of incresed tokens.
                fresh_k_part = fresh_k[q_start:q_end].contiguous()
                fresh_v_part = fresh_v[q_start:q_end].contiguous()
                torch_npu.npu_fused_infer_attention_score.out(
                    query=q_part.unsqueeze(0),
                    key=fresh_k_part.unsqueeze(0),
                    value=fresh_v_part.unsqueeze(0),
                    input_layout="BSND",
                    sparse_mode=3,
                    atten_mask=self.atten_mask,
                    scale=HEAD_DIM**-0.5,
                    next_tokens=0,
                    actual_seq_lengths=[q_len],
                    actual_seq_lengths_kv=[q_len],
                    num_heads=N_Q,
                    num_key_value_heads=N_KV,
                    softmax_lse_flag=True,
                    out=[fia_out, fia_lse],
                )
                part_out, part_lse = self._normalize_bsnd_fia_outputs(fia_out, fia_lse)
                self._merge_fia_part(
                    acc_out,
                    acc_lse,
                    part_out,
                    part_lse,
                    weighted_part,
                    acc_initialized,
                )
                output[q_start:q_end].copy_(acc_out)
                del fia_out, fia_lse, part_out, part_lse
                del acc_out, acc_lse, weighted_part

            return output

        # No K/V cache dequantization is needed, use the standard fused attention kernel.
        return torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k.contiguous(),
            value=v.contiguous(),
            input_layout="TND",
            sparse_mode=3,
            atten_mask=self.atten_mask,
            scale=HEAD_DIM**-0.5,
            next_tokens=0,
            actual_seq_lengths=self.infer_state.b1_cu_q_seq_len_cpu,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_heads=N_Q,
            num_key_value_heads=N_KV,
        )[0]


@dataclasses.dataclass
class PagedFa3Int8KVAscendDecodeAttState(PagedFa3DecodeAttState):

    def _normal_decode_att(
        self,
        q: torch.Tensor,
        k: tuple[torch.Tensor, torch.Tensor],
        v: tuple[torch.Tensor, torch.Tensor],
        att_control: AttControl,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        if att_control.use_att_sink:
            raise NotImplementedError(
                "Ascend INT8 KV decode does not support attention sink"
            )
        if att_control.use_sliding_window:
            raise NotImplementedError(
                "Ascend paged INT8 KV decode does not support sliding-window attention"
            )
        return super()._normal_decode_att(q, k, v, att_control, alloc_func)

    def _prepare_npu_kv_cache(
        self,
        k: tuple[torch.Tensor, torch.Tensor],
        v: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
        k_cache, k_scale = k
        v_cache, v_scale = v
        N_KV, HEAD_DIM = k_cache.shape[-2:]
        k_cache = k_cache.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
        v_cache = v_cache.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
        return (
            k_cache,
            v_cache,
            N_KV,
            {
                "key_antiquant_scale": k_scale,
                "value_antiquant_scale": v_scale,
                "key_antiquant_mode": NPU_PAGED_PER_TOKEN_ANTIQUANT_MODE,
                "value_antiquant_mode": NPU_PAGED_PER_TOKEN_ANTIQUANT_MODE,
            },
        )


@register_att_backend(
    name="paged_fa3",
    category="standard",
    kv_types=("int8kv",),
    platforms=("maca",),
    validate_name="fa3",
)
class PagedFa3Int8KVMacaAttBackend(PagedFa3AttBackend):

    # Group fixed native-MTP verify rows to reuse their common INT8 KV prefix.
    supports_grouped_eagle_extend = True

    def __init__(self, model, page_size=None):
        super().__init__(model=model, page_size=page_size)
        self.quant_group_size = get_env_start_args().llm_kv_quant_group_size
        if maca_flash_attn_func is None or maca_flash_attn_varlen_func is None:
            raise RuntimeError(
                "MetaX INT8 KV prefill requires flash_attn_func and "
                "flash_attn_varlen_func from the flash_attn package"
            )
        logger.warning(
            "MetaX flash_attn_with_kvcache_dequant is not used: it is numerically "
            "wrong except headdim=128 and requires PAGE_SIZE %% 256 == 0. "
            "Decode uses the fused Triton int8kv kernel."
        )

    def create_att_prefill_state(
        self, infer_state: "InferStateInfo"
    ) -> "PagedFa3Int8KVMacaPrefillAttState":
        return PagedFa3Int8KVMacaPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(
        self, infer_state: "InferStateInfo"
    ) -> "PagedFa3Int8KVMacaDecodeAttState":
        return PagedFa3Int8KVMacaDecodeAttState(backend=self, infer_state=infer_state)


class PagedFa3Int8KVMacaPrefillAttState(PagedFa3PrefillAttState):
    request_slices: tuple = None
    max_prefix_len: int = 0
    dequant_chunk_tokens: int = DEFAULT_MACA_DEQUANT_CHUNK_TOKENS
    oneshot_max_bytes: int = DEFAULT_MACA_ONESHOT_MAX_BYTES
    force_chunked_prefix: bool = False

    def init_state(self) -> None:
        infer = self.infer_state
        self.prefix_total_token_num = infer.total_token_num - infer.input_ids.shape[0]
        self.cu_seqlens_q = infer.b1_cu_q_seq_len.int()
        # Fresh / packed K is aligned to Q lengths; prefix KV is gathered separately.
        self.cu_seqlens_k = None
        # Prefix cache is gathered via req_to_token, not a FA3 page table.
        self.page_table = None
        self.atten_mask = None
        self.request_slices = None
        self.max_prefix_len = 0
        self.dequant_chunk_tokens = int(
            os.getenv(
                "LIGHTLLM_MACA_INT8KV_DEQUANT_CHUNK_TOKENS",
                str(DEFAULT_MACA_DEQUANT_CHUNK_TOKENS),
            )
        )
        if self.dequant_chunk_tokens <= 0:
            raise ValueError(
                "LIGHTLLM_MACA_INT8KV_DEQUANT_CHUNK_TOKENS must be greater than zero"
            )
        self.oneshot_max_bytes = int(
            os.getenv(
                "LIGHTLLM_MACA_INT8KV_ONESHOT_MAX_BYTES",
                str(DEFAULT_MACA_ONESHOT_MAX_BYTES),
            )
        )
        self.force_chunked_prefix = os.getenv(
            "LIGHTLLM_MACA_INT8KV_PREFIX_ONESHOT", "1"
        ) == "0"
        if self.prefix_total_token_num == 0:
            return

        q_cu = infer.b1_cu_q_seq_len.detach().to("cpu", non_blocking=True)
        prefix = infer.b_ready_cache_len.detach().to("cpu", non_blocking=True)
        req_ids = infer.b_req_idx.detach().to("cpu", non_blocking=True)
        torch.cuda.current_stream().synchronize()
        q_cu = q_cu.tolist()
        prefix = prefix.tolist()
        req_ids = req_ids.tolist()
        slices = []
        max_prefix_len = 0
        for i, req_id in enumerate(req_ids):
            q_start, q_end = int(q_cu[i]), int(q_cu[i + 1])
            prefix_len = int(prefix[i])
            if q_end <= q_start or prefix_len < 0:
                raise ValueError(
                    f"Invalid prefill lengths: q_len={q_end - q_start}, prefix_len={prefix_len}"
                )
            max_prefix_len = max(max_prefix_len, prefix_len)
            slices.append((q_start, q_end, int(req_id), prefix_len))
        self.request_slices = tuple(slices)
        self.max_prefix_len = max_prefix_len

    def _normal_prefill_att(
        self,
        q: torch.Tensor,
        k: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        v: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        att_control: AttControl,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        if att_control.use_att_sink:
            raise NotImplementedError("MetaX FlashAttention does not support attention sinks")
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        fresh_k, k_cache, k_scale = k
        fresh_v, v_cache, v_scale = v
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

        if self.prefix_total_token_num == 0:
            return maca_flash_attn_varlen_func(
                q=q,
                k=fresh_k,
                v=fresh_v,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_q,
                max_seqlen_q=self.infer_state.max_q_seq_len,
                max_seqlen_k=self.infer_state.max_q_seq_len,
                softmax_scale=sm_scale,
                causal=True,
                window_size=window_size,
                softcap=0.0,
            )

        if att_control.use_sliding_window:
            raise NotImplementedError(
                "MetaX INT8 KV chunked prefix prefill does not support sliding-window attention"
            )
        return self._chunked_prefix_prefill_att(
            q=q,
            fresh_k=fresh_k,
            fresh_v=fresh_v,
            k_cache=k_cache,
            k_scale=k_scale,
            v_cache=v_cache,
            v_scale=v_scale,
            sm_scale=sm_scale,
            alloc_func=alloc_func,
        )

    def _chunked_prefix_prefill_att(
        self,
        q: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        k_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_cache: torch.Tensor,
        v_scale: torch.Tensor,
        sm_scale: float,
        alloc_func: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        n_q, head_dim = q.shape[-2:]
        n_kv = k_cache.shape[-2]
        group_size = self.backend.quant_group_size
        total_kv_tokens = int(self.prefix_total_token_num) + int(q.shape[0])
        oneshot_bytes = total_kv_tokens * n_kv * head_dim * q.element_size() * 2
        if (
            not self.force_chunked_prefix
            and self.oneshot_max_bytes > 0
            and oneshot_bytes <= self.oneshot_max_bytes
        ):
            return self._oneshot_prefix_prefill_att(
                q=q,
                fresh_k=fresh_k,
                fresh_v=fresh_v,
                k_cache=k_cache,
                k_scale=k_scale,
                v_cache=v_cache,
                v_scale=v_scale,
                sm_scale=sm_scale,
                alloc_func=alloc_func,
            )
        chunk_tokens = min(self.dequant_chunk_tokens, max(self.max_prefix_len, 1))
        req_to_token = self.infer_state.req_manager.req_to_token_indexs
        acc_out = alloc_func(q.shape, dtype=torch.float32, device=q.device)
        acc_lse = alloc_func((q.shape[0], n_q, 1), dtype=torch.float32, device=q.device)
        weighted_part = alloc_func(
            (self.infer_state.max_q_seq_len, n_q, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        n_req = len(self.request_slices)
        k_scratch = alloc_func(
            (n_req * chunk_tokens, n_kv, head_dim), dtype=q.dtype, device=q.device
        )
        v_scratch = alloc_func(
            (n_req * chunk_tokens, n_kv, head_dim), dtype=q.dtype, device=q.device
        )
        acc_initialized = [False] * n_req

        for chunk_start in range(0, self.max_prefix_len, chunk_tokens):
            self._prefix_chunk_att(
                q=q,
                k_cache=k_cache,
                k_scale=k_scale,
                v_cache=v_cache,
                v_scale=v_scale,
                req_to_token=req_to_token,
                chunk_start=chunk_start,
                chunk_tokens=chunk_tokens,
                group_size=group_size,
                sm_scale=sm_scale,
                k_scratch=k_scratch,
                v_scratch=v_scratch,
                acc_out=acc_out,
                acc_lse=acc_lse,
                weighted_part=weighted_part,
                acc_initialized=acc_initialized,
            )

        fresh_out, fresh_lse = self._flash_varlen_with_lse(
            q,
            fresh_k,
            fresh_v,
            self.cu_seqlens_q,
            self.cu_seqlens_q,
            self.infer_state.max_q_seq_len,
            self.infer_state.max_q_seq_len,
            sm_scale,
            causal=True,
        )
        for i, (q_start, q_end, _, _) in enumerate(self.request_slices):
            q_len = q_end - q_start
            _merge_attn_lse(
                acc_out[q_start:q_end],
                acc_lse[q_start:q_end],
                fresh_out[q_start:q_end].float(),
                fresh_lse[q_start:q_end],
                weighted_part[:q_len],
                acc_initialized[i],
            )
        output = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        output.copy_(acc_out)
        return output

    def _oneshot_prefix_prefill_att(
        self,
        q: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        k_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_cache: torch.Tensor,
        v_scale: torch.Tensor,
        sm_scale: float,
        alloc_func: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        n_kv, head_dim = k_cache.shape[-2], q.shape[-1]
        group_size = self.backend.quant_group_size
        req_to_token = self.infer_state.req_manager.req_to_token_indexs
        total_kv = int(self.prefix_total_token_num) + int(q.shape[0])
        k_pack = alloc_func((total_kv, n_kv, head_dim), dtype=q.dtype, device=q.device)
        v_pack = alloc_func((total_kv, n_kv, head_dim), dtype=q.dtype, device=q.device)
        cu_k = [0]
        kv_off = 0
        max_kv = 0
        for q_start, q_end, req_id, prefix_len in self.request_slices:
            q_len = q_end - q_start
            if prefix_len > 0:
                gather_dequant_int8kv(
                    k_cache,
                    k_scale,
                    v_cache,
                    v_scale,
                    req_to_token[req_id, :prefix_len],
                    k_pack[kv_off : kv_off + prefix_len],
                    v_pack[kv_off : kv_off + prefix_len],
                    group_size,
                )
            k_pack[kv_off + prefix_len : kv_off + prefix_len + q_len].copy_(fresh_k[q_start:q_end])
            v_pack[kv_off + prefix_len : kv_off + prefix_len + q_len].copy_(fresh_v[q_start:q_end])
            kv_off += prefix_len + q_len
            cu_k.append(kv_off)
            max_kv = max(max_kv, prefix_len + q_len)

        infer = self.infer_state
        cu_seqlens_k = getattr(infer, "b1_cu_kv_seq_len", None)
        if cu_seqlens_k is None or int(cu_seqlens_k[-1]) != kv_off:
            cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=q.device)
        else:
            cu_seqlens_k = cu_seqlens_k.int()
        return maca_flash_attn_varlen_func(
            q=q,
            k=k_pack,
            v=v_pack,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=infer.max_q_seq_len,
            max_seqlen_k=max(max_kv, int(getattr(infer, "max_kv_seq_len", 0) or 0)),
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
        )

    def _prefix_chunk_att(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_cache: torch.Tensor,
        v_scale: torch.Tensor,
        req_to_token: torch.Tensor,
        chunk_start: int,
        chunk_tokens: int,
        group_size: int,
        sm_scale: float,
        k_scratch: torch.Tensor,
        v_scratch: torch.Tensor,
        acc_out: torch.Tensor,
        acc_lse: torch.Tensor,
        weighted_part: torch.Tensor,
        acc_initialized: list[bool],
    ) -> None:
        q_parts = []
        idx_parts = []
        cu_q = [0]
        cu_k = [0]
        active = []
        max_q = 0
        max_k = 0
        for i, (q_start, q_end, req_id, prefix_len) in enumerate(self.request_slices):
            if chunk_start >= prefix_len:
                continue
            chunk_end = min(prefix_len, chunk_start + chunk_tokens)
            kv_len = chunk_end - chunk_start
            q_len = q_end - q_start
            q_parts.append(q[q_start:q_end])
            idx_parts.append(req_to_token[req_id, chunk_start:chunk_end])
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + kv_len)
            active.append((i, q_start, q_end, q_len))
            max_q = max(max_q, q_len)
            max_k = max(max_k, kv_len)
        if not active:
            return

        indices = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts)
        gather_dequant_int8kv(
            k_cache,
            k_scale,
            v_cache,
            v_scale,
            indices,
            k_scratch[: cu_k[-1]],
            v_scratch[: cu_k[-1]],
            group_size,
        )
        if len(active) == 1:
            i, q_start, q_end, q_len = active[0]
            part_out, part_lse = self._flash_with_lse(
                q_parts[0],
                k_scratch[: cu_k[-1]],
                v_scratch[: cu_k[-1]],
                causal=False,
                sm_scale=sm_scale,
            )
            _merge_attn_lse(
                acc_out[q_start:q_end],
                acc_lse[q_start:q_end],
                part_out.float(),
                part_lse,
                weighted_part[:q_len],
                acc_initialized[i],
            )
            acc_initialized[i] = True
            return

        q_pack = torch.cat(q_parts)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=q.device)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=q.device)
        part_out, part_lse = self._flash_varlen_with_lse(
            q_pack,
            k_scratch[: cu_k[-1]],
            v_scratch[: cu_k[-1]],
            cu_q_t,
            cu_k_t,
            max_q,
            max_k,
            sm_scale,
            causal=False,
        )
        offset = 0
        for i, q_start, q_end, q_len in active:
            _merge_attn_lse(
                acc_out[q_start:q_end],
                acc_lse[q_start:q_end],
                part_out[offset : offset + q_len].float(),
                part_lse[offset : offset + q_len],
                weighted_part[:q_len],
                acc_initialized[i],
            )
            acc_initialized[i] = True
            offset += q_len

    @staticmethod
    def _flash_with_lse(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        sm_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len, n_q, _ = q.shape
        result = maca_flash_attn_func(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            softmax_scale=sm_scale,
            causal=causal,
            return_attn_probs=True,
            softcap=0.0,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("MetaX flash_attn_func did not return softmax LSE")
        out, lse = result[0].squeeze(0), result[1]
        return out, PagedFa3Int8KVMacaPrefillAttState._normalize_lse(lse, q_len, n_q)

    @staticmethod
    def _flash_varlen_with_lse(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        sm_scale: float,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = maca_flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=sm_scale,
            causal=causal,
            return_attn_probs=True,
            softcap=0.0,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("MetaX flash_attn_varlen_func did not return softmax LSE")
        n_q = q.shape[1]
        return result[0], PagedFa3Int8KVMacaPrefillAttState._normalize_lse(result[1], q.shape[0], n_q)

    @staticmethod
    def _normalize_lse(lse: torch.Tensor, q_len: int, n_q: int) -> torch.Tensor:
        if lse.dim() == 3 and lse.shape[0] == 1:
            lse = lse.squeeze(0)
        if lse.dim() != 2:
            raise RuntimeError(f"Unexpected softmax_lse shape {tuple(lse.shape)}")
        if lse.shape[0] == n_q and lse.shape[-1] >= q_len:
            lse = lse[:, :q_len].transpose(0, 1)
        elif lse.shape[0] >= q_len and lse.shape[1] == n_q:
            lse = lse[:q_len]
        else:
            raise RuntimeError(
                f"Unexpected softmax_lse shape {tuple(lse.shape)} for q_len={q_len}, n_q={n_q}"
            )
        return lse.unsqueeze(-1).contiguous()


class PagedFa3Int8KVMacaDecodeAttState(PagedFa3DecodeAttState):

    supports_noncausal_block = True

    def _normal_decode_att(
        self,
        q: torch.Tensor,
        k: tuple[torch.Tensor, torch.Tensor],
        v: tuple[torch.Tensor, torch.Tensor],
        att_control: AttControl,
        alloc_func: Callable[..., torch.Tensor] = torch.empty,
    ) -> torch.Tensor:
        if att_control.use_alibi:
            raise NotImplementedError("MetaX INT8 KV decode does not support ALiBi")
        if att_control.use_att_sink:
            raise NotImplementedError("MetaX FlashAttention does not support attention sinks")
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        k_cache, k_scale = k
        v_cache, v_scale = v
        head_dim = k_cache.shape[-1]
        page_size = self.backend.page_size
        group_size = self.backend.quant_group_size

        att_batch_size = self.page_table.shape[0] if self.causal else self.b_block_req_idx.shape[0]
        expected_tokens = att_batch_size * self.decode_max_q_seq_len
        if q.shape[0] > expected_tokens or (q.shape[0] != expected_tokens and not self.has_partial_block):
            raise ValueError(
                "Unexpected MetaX INT8 KV decode query shape: "
                f"q tokens={q.shape[0]}, batch={att_batch_size}, "
                f"q_len={self.decode_max_q_seq_len}"
            )

        if q.shape[0] < expected_tokens:
            # Only the final graph HOLD group may be partial. Pad unused rows;
            # real request blocks retain the same fixed-width layout.
            padded_q = alloc_func((expected_tokens, *q.shape[1:]), dtype=q.dtype, device=q.device)
            padded_q.zero_()
            padded_q[:q.shape[0]].copy_(q)
        else:
            padded_q = q
        q_bshd = padded_q.view(att_batch_size, self.decode_max_q_seq_len, q.shape[1], q.shape[2])
        from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.maca_int8kv_flash_decoding import (
            int8kv_flash_decode,
        )

        output = int8kv_flash_decode(
            q=q_bshd,
            k=k_cache,
            k_scale=k_scale,
            v=v_cache,
            v_scale=v_scale,
            cache_seqlens=self.b_att_seq_len,
            page_table=(self.page_table if self.causal else self.backend.model.req_manager.req_to_token_indexs),
            token_req_indices=None if self.causal else self.b_block_req_idx,
            page_size=page_size,
            sm_scale=1.0 / (head_dim ** 0.5),
            causal=self.causal,
            sliding_window=window_size,
            quant_group_size=group_size,
            max_kv_len=int(self.infer_state.max_kv_seq_len if self.causal else self.block_max_kv_len),
            alloc_func=alloc_func,
        )
        return output.reshape(-1, q.shape[1], q.shape[2])[:q.shape[0]]
