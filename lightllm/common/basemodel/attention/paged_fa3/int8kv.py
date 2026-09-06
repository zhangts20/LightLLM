import dataclasses
import os
from typing import TYPE_CHECKING, Any, Callable

import torch

from lightllm.platform.base.attention import register_att_backend
from lightllm.utils.log_utils import init_logger

from ..base_att import AttControl
from .fp import PagedFa3AttBackend, PagedFa3DecodeAttState, PagedFa3PrefillAttState

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


NPU_PAGED_PER_TOKEN_ANTIQUANT_MODE = 4
DEFAULT_DEQUANT_CHUNK_TOKENS = 65536
DEFAULT_DEQUANT_RESERVE_MIB = 512
logger = init_logger(__name__)


@register_att_backend(
    name="paged_fa3",
    category="standard",
    kv_types=("int8kv",),
    platforms=("ascend",),
    validate_name="fa3",
)
class PagedFa3Int8KVAttBackend(PagedFa3AttBackend):

    def create_att_prefill_state(
        self, infer_state: "InferStateInfo"
    ) -> "PagedFa3Int8KVPrefillAttState":
        return PagedFa3Int8KVPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(
        self, infer_state: "InferStateInfo"
    ) -> "PagedFa3Int8KVDecodeAttState":
        return PagedFa3Int8KVDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class PagedFa3Int8KVPrefillAttState(PagedFa3PrefillAttState):
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
            self.infer_state.prefix_total_token_num != 0
            and self.infer_state.max_q_seq_len == 1
        ):
            PagedFa3PrefillAttState.init_state(self)
            self.use_paged_int8 = True
            return

        self.atten_mask = self.backend.get_causal_attn_mask(
            self.infer_state.input_ids.device
        )

        if self.infer_state.prefix_total_token_num == 0:
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
                self.infer_state.prefix_total_token_num != 0
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
class PagedFa3Int8KVDecodeAttState(PagedFa3DecodeAttState):

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
