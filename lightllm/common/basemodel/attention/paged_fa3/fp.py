import dataclasses
from typing import Any

import torch
import triton
import triton.language as tl
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from lightllm.utils.envs_utils import get_env_start_args, get_page_size
from lightllm.common.basemodel.triton_kernel.fa3_utils import page_table_copy
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor
from lightllm.platform.base.attention import register_att_backend
from .graph_utils import weak_ref_tensor

try:
    from flash_attn import flash_attn_varlen_func as maca_flash_attn_varlen_func
    from flash_attn import flash_attn_with_kvcache as maca_flash_attn_with_kvcache
except ImportError:
    maca_flash_attn_varlen_func = None
    maca_flash_attn_with_kvcache = None


@triton.jit
def _gather_block_kv_kernel(
    K, V, OUT_K, OUT_V, REQ_TO_TOKEN, REQ_IDS, SEQ_LENS, CU_K,
    REQ_STRIDE: tl.constexpr, K_STRIDE: tl.constexpr, V_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr, V_HEAD_STRIDE: tl.constexpr,
    K_DIM_STRIDE: tl.constexpr, V_DIM_STRIDE: tl.constexpr,
    HEADS: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr,
):
    batch = tl.program_id(1)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // (HEADS * DIM)
    feature = offsets % (HEADS * DIM)
    seq_len = tl.load(SEQ_LENS + batch)
    req = tl.load(REQ_IDS + batch)
    start = tl.load(CU_K + batch)
    valid = token < seq_len
    slot = tl.load(REQ_TO_TOKEN + req * REQ_STRIDE + token, mask=valid, other=0)
    kval = tl.load(K + slot * K_STRIDE + feature // DIM * K_HEAD_STRIDE
                   + feature % DIM * K_DIM_STRIDE, mask=valid, other=0)
    vval = tl.load(V + slot * V_STRIDE + feature // DIM * V_HEAD_STRIDE
                   + feature % DIM * V_DIM_STRIDE, mask=valid, other=0)
    output_offset = (start + token) * HEADS * DIM + feature
    tl.store(OUT_K + output_offset, kval, mask=valid)
    tl.store(OUT_V + output_offset, vval, mask=valid)


@register_att_backend(name="paged_fa3", category="standard", platforms=("ascend", "cuda", "maca",), validate_name="fa3")
class PagedFa3AttBackend(BaseAttBackend):

    def __init__(self, model, page_size=None):
        super().__init__(model=model)
        self.page_size = page_size or get_page_size()
        self.is_maca = get_env_start_args().hardware_platform == "maca"
        if self.is_maca:
            if self.page_size % 16 != 0:
                raise ValueError(
                    "MetaX FlashAttention requires PAGE_SIZE to be a multiple "
                    f"of 16, but got PAGE_SIZE={self.page_size}"
                )
            if (
                maca_flash_attn_varlen_func is None
                or maca_flash_attn_with_kvcache is None
            ):
                raise RuntimeError(
                    "MetaX paged FlashAttention requires flash_attn_varlen_func "
                    "and flash_attn_with_kvcache from the flash_attn package"
                )
        self.get_page_table_buffer()

    def get_page_table_buffer(self):
        model = self.model
        if not hasattr(self, "_shared_page_table_buffer"):
            shared_len = model.graph_max_batch_size * triton.cdiv(model.graph_max_len_in_batch, self.page_size)
            self._shared_page_table_buffer = [
                torch.empty(shared_len, dtype=torch.int32).to(get_current_device_id()),
                torch.empty(shared_len, dtype=torch.int32).to(get_current_device_id()),
            ]
        return self._shared_page_table_buffer

    def get_causal_attn_mask(self, device):
        if not hasattr(self, "_causal_attn_mask"):
            self._causal_attn_mask = torch.triu(
                torch.ones((2048, 2048), dtype=torch.int8, device=device), diagonal=1
            )
        return self._causal_attn_mask

    def get_decode_seq_len_cpu_buffers(self, min_len: int):
        """Pinned CPU int32 buffers reused for npu_fused_infer_attention_score list args."""
        model = self.model
        cap = max(min_len, model.graph_max_batch_size)
        if not hasattr(self, "_decode_seq_len_cpu_q") or self._decode_seq_len_cpu_q.shape[0] < min_len:
            self._decode_seq_len_cpu_q = torch.empty(cap, dtype=torch.int32, pin_memory=True)
            self._decode_seq_len_cpu_kv = torch.empty(cap, dtype=torch.int32, pin_memory=True)
        return self._decode_seq_len_cpu_q, self._decode_seq_len_cpu_kv

    def create_att_prefill_state(self, infer_state):
        return PagedFa3PrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return PagedFa3DecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class PagedFa3PrefillAttState(BasePrefillAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    atten_mask: torch.Tensor = None

    def init_state(self):
        self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
        self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()
        table_len = triton.cdiv(self.infer_state.max_kv_seq_len, self.backend.page_size)
        self.page_table = torch.empty(
            (self.infer_state.batch_size, table_len),
            dtype=torch.int32,
            device=self.infer_state.input_ids.device,
        )
        page_table_copy(
            page_table=self.page_table,
            req_to_token_indexs=self.infer_state.req_manager.req_to_token_indexs,
            b_req_idx=self.infer_state.b_req_idx,
            page_size=self.backend.page_size,
        )
        self.atten_mask = self.backend.get_causal_attn_mask(self.infer_state.input_ids.device)

    def prefill_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert att_control.use_alibi is False
        return self._normal_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _normal_prefill_att(self, q, k, v, att_control: AttControl, alloc_func=torch.empty):
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        if att_control.use_att_sink:
            sink_weight = att_control.sink_weight
        else:
            sink_weight = None

        sm_scale = 1.0 / (q.shape[-1] ** 0.5)

        if self.backend.is_maca:
            if sink_weight is not None:
                raise NotImplementedError(
                    "MetaX FlashAttention does not support attention sinks"
                )
            return maca_flash_attn_varlen_func(
                q=q,
                k=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                v=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
                block_table=self.page_table,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k=self.cu_seqlens_k,
                max_seqlen_q=self.infer_state.max_q_seq_len,
                max_seqlen_k=self.infer_state.max_kv_seq_len,
                softmax_scale=sm_scale,
                causal=True,
                window_size=window_size,
                softcap=0.0,
            )
        elif q.device.type == "npu":
            import torch_npu

            N_Q, HEAD_DIM = q.shape[-2:]
            N_KV = k.shape[-2]
            key = k.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
            value = v.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
            return torch_npu.npu_fused_infer_attention_score(
                query=q,
                key=key,
                value=value,
                input_layout="TND",
                sparse_mode=3,
                atten_mask=self.atten_mask,
                scale=sm_scale,
                next_tokens=0,
                actual_seq_lengths=self.infer_state.b1_cu_q_seq_len_cpu,
                actual_seq_lengths_kv=self.infer_state.b_cu_kv_seq_len_cpu,
                num_heads=N_Q,
                num_key_value_heads=N_KV,
                block_table=self.page_table,
                block_size=self.backend.page_size,
            )[0]
        else:
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                v_cache=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
                page_table=self.page_table,
                cache_seqlens=self.infer_state.b_seq_len,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k_new=self.cu_seqlens_k,
                max_seqlen_q=self.infer_state.max_q_seq_len,
                softmax_scale=sm_scale,
                causal=True,
                window_size=window_size,
                softcap=0.0,
                k_descale=None,
                v_descale=None,
                return_softmax_lse=False,
                sinks=sink_weight,
            )


@dataclasses.dataclass
class PagedFa3DecodeAttState(BaseDecodeAttState):
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None
    b_att_seq_len: torch.Tensor = None
    decode_max_q_seq_len: int = None
    use_mtp_bnsd: bool = False

    causal: bool = True
    has_partial_block: bool = False
    b_block_req_idx: torch.Tensor = None
    block_max_kv_len: int = None

    def init_state(self):
        args = get_env_start_args()
        model = self.backend.model
        is_block_mode = args.mtp_mode in ("dspark", "dflash")
        # Proposers and graph capture use model-local widths: recurrent and
        # chained drafts use one row, block drafts mtp_step, main mtp_step + 1.
        args_mtp_step = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model)
        query_group_size = getattr(self.infer_state, "decode_query_group_size", 1)
        if query_group_size > 1:
            assert model.is_mtp_draft_model and not is_block_mode
            args_mtp_step = query_group_size - 1
        self.causal = True
        if is_block_mode:
            if self.backend.uses_dynamic_spec_verify_layout():
                raise NotImplementedError(
                    "paged_fa3 block verification requires mtp_dynamic_verify=False"
                )
            self.causal = self.backend.uses_causal_attention()
            if not self.causal and (
                self.infer_state.input_ids.device.type == "npu"
                or (type(self)._normal_decode_att is not PagedFa3DecodeAttState._normal_decode_att
                    and not getattr(self, "supports_noncausal_block", False))
            ):
                # Require an explicit noncausal implementation in specialized
                # backends; Ascend per-row BNSD remains causal-only.
                raise NotImplementedError(
                    "paged_fa3 noncausal block decode requires a supported CUDA or MACA backend"
                )

        mtp_size = args_mtp_step + 1
        rows = self.infer_state.batch_size
        self.has_partial_block = rows % mtp_size != 0
        if not is_block_mode:
            assert not self.has_partial_block
        # Fixed-layout graph/TPSP padding appends HOLD rows. A final partial
        # group belongs only to that padding, never to a real request. Keep
        # its true query length so capture need not round graph_max_batch_size.
        att_batch_size = triton.cdiv(rows, mtp_size)
        if args_mtp_step > 0:
            b_q_seq_len = torch.full(
                (att_batch_size,), mtp_size, dtype=torch.int32,
                device=self.infer_state.b_seq_len.device,
            )
            if self.has_partial_block:
                b_q_seq_len[-1:] = rows % mtp_size
            last_rows = (torch.arange(att_batch_size, device=b_q_seq_len.device) * mtp_size + mtp_size - 1).clamp_max(rows - 1)
            b_kv_seq_len = self.infer_state.b_seq_len.index_select(0, last_rows)
            b1_cu_q_seq_len, b1_cu_kv_seq_len = gen_cumsum_pad0_tensor(b_q_seq_len, b_kv_seq_len)
            self.cu_seqlens_q = b1_cu_q_seq_len.int()
            self.cu_seqlens_k = b1_cu_kv_seq_len.int()
        else:
            self.cu_seqlens_q = self.infer_state.b1_cu_q_seq_len.int()
            self.cu_seqlens_k = self.infer_state.b1_cu_kv_seq_len.int()

        self.use_mtp_bnsd = args_mtp_step > 0 and self.infer_state.input_ids.device.type == "npu"
        page_table_batch_size = rows if self.use_mtp_bnsd else att_batch_size
        model = self.backend.model
        if not self.causal:
            # Scratch slots may start in the middle of a logical page while
            # physically starting a new page. Do not divide their token IDs by
            # page_size: preserve the exact prefix + scratch mapping.
            self.b_block_req_idx = (
                self.infer_state.b_req_idx.index_select(0, last_rows)
                if args_mtp_step > 0 else self.infer_state.b_req_idx
            )
            self.b_att_seq_len = b_kv_seq_len.contiguous() if args_mtp_step > 0 else self.infer_state.b_seq_len
            self.decode_max_q_seq_len = mtp_size
            self.block_max_kv_len = self.infer_state.max_kv_seq_len
            if rows <= model.graph_max_batch_size and self.block_max_kv_len <= model.graph_max_len_in_batch:
                self.block_max_kv_len = model.graph_max_len_in_batch
            return
        table_len = triton.cdiv(self.infer_state.max_kv_seq_len, self.backend.page_size)
        if (
            self.infer_state.batch_size <= model.graph_max_batch_size
            and self.infer_state.max_kv_seq_len <= model.graph_max_len_in_batch
        ):
            page_buffer = self.backend.get_page_table_buffer()
            shared_table_len = triton.cdiv(model.graph_max_len_in_batch, self.backend.page_size)
            self.page_table = page_buffer[self.infer_state.microbatch_index][
                : page_table_batch_size * shared_table_len
            ].reshape(page_table_batch_size, shared_table_len)
        else:
            self.page_table = torch.empty(
                (page_table_batch_size, table_len),
                dtype=torch.int32,
                device=self.infer_state.input_ids.device,
            )

        if args_mtp_step > 0:
            page_table_req_idx = (
                self.infer_state.b_req_idx
                if self.use_mtp_bnsd
                else self.infer_state.b_req_idx.index_select(0, last_rows)
            )
            page_table_copy(
                page_table=self.page_table[:, :table_len],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=page_table_req_idx,
                page_size=self.backend.page_size,
            )
            self.b_att_seq_len = b_kv_seq_len.contiguous()
            self.decode_max_q_seq_len = args_mtp_step + 1
        else:
            page_table_copy(
                page_table=self.page_table[:, :table_len],
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                b_req_idx=self.infer_state.b_req_idx,
                page_size=self.backend.page_size,
            )
            self.b_att_seq_len = self.infer_state.b_seq_len
            self.decode_max_q_seq_len = 1

    def decode_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert att_control.use_alibi is False
        return self._normal_decode_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _prepare_npu_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
        N_KV, HEAD_DIM = k.shape[-2:]
        k = k.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
        v = v.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
        return k, v, N_KV, {}

    def _block_decode_att(self, q, k, v, window_size, sink_weight, alloc_func):
        req_to_token = self.backend.model.req_manager.req_to_token_indexs
        if not self.backend.is_maca:
            # CUDA FA3 supports token-sized pages directly, avoiding a KV copy.
            token_table = req_to_token[self.b_block_req_idx, :self.block_max_kv_len].int().contiguous()
            return flash_attn_with_kvcache(
                q=q, k_cache=k.unsqueeze(1), v_cache=v.unsqueeze(1),
                page_table=token_table, cache_seqlens=self.b_att_seq_len,
                cu_seqlens_q=self.cu_seqlens_q, cu_seqlens_k_new=self.cu_seqlens_k,
                max_seqlen_q=self.decode_max_q_seq_len,
                softmax_scale=q.shape[-1] ** -0.5, causal=False,
                window_size=window_size, softcap=0.0,
                k_descale=None, v_descale=None, return_softmax_lse=False, sinks=sink_weight,
            )
        if sink_weight is not None:
            raise NotImplementedError("MetaX FlashAttention does not support attention sinks")
        # One packed gather per block, shared by every query in the block.
        # Allocate an upper bound; GPU cumulative lengths delimit actual KV.
        # No host reads or data-dependent allocations during graph capture.
        shape = (self.b_block_req_idx.shape[0] * self.block_max_kv_len, k.shape[1], k.shape[2])
        packed_k = alloc_func(shape, dtype=k.dtype, device=k.device)
        packed_v = alloc_func(shape, dtype=v.dtype, device=v.device)
        _gather_block_kv_kernel[
            (triton.cdiv(self.block_max_kv_len * k.shape[1] * k.shape[2], 256), self.b_block_req_idx.shape[0])
        ](
            k, v, packed_k, packed_v, req_to_token, self.b_block_req_idx,
            self.b_att_seq_len, self.cu_seqlens_k,
            req_to_token.stride(0), k.stride(0), v.stride(0),
            k.stride(1), v.stride(1), k.stride(2), v.stride(2),
            k.shape[1], k.shape[2], 256,
        )
        return maca_flash_attn_varlen_func(
            q=q, k=packed_k, v=packed_v,
            cu_seqlens_q=self.cu_seqlens_q, cu_seqlens_k=self.cu_seqlens_k,
            max_seqlen_q=self.decode_max_q_seq_len, max_seqlen_k=self.block_max_kv_len,
            softmax_scale=q.shape[-1] ** -0.5, causal=False,
            window_size=window_size, softcap=0.0,
        )

    def _normal_decode_att(self, q, k, v, att_control: AttControl, alloc_func=torch.empty):
        if att_control.use_sliding_window:
            window_size = att_control.sliding_window
        else:
            window_size = (-1, -1)

        if att_control.use_att_sink:
            sink_weight = att_control.sink_weight
        else:
            sink_weight = None

        if not self.causal:
            if q.shape[0] != self.infer_state.batch_size:
                raise ValueError("Unexpected MetaX decode query shape for noncausal block")
            return self._block_decode_att(q, k, v, window_size, sink_weight, alloc_func)

        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        if self.backend.is_maca:
            if sink_weight is not None:
                raise NotImplementedError(
                    "MetaX FlashAttention does not support attention sinks"
                )

            if self.has_partial_block:
                # BSHD kvcache requires a uniform query width; varlen handles
                # the short trailing HOLD group without adding query rows.
                return maca_flash_attn_varlen_func(
                    q=q,
                    k=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                    v=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
                    block_table=self.page_table,
                    cu_seqlens_q=self.cu_seqlens_q,
                    cu_seqlens_k=self.cu_seqlens_k,
                    max_seqlen_q=self.decode_max_q_seq_len,
                    max_seqlen_k=self.infer_state.max_kv_seq_len,
                    softmax_scale=sm_scale,
                    causal=self.causal,
                    window_size=window_size,
                    softcap=0.0,
                )

            att_batch_size = self.page_table.shape[0]
            expected_tokens = att_batch_size * self.decode_max_q_seq_len
            if q.shape[0] != expected_tokens:
                raise ValueError(
                    "Unexpected MetaX decode query shape: "
                    f"q tokens={q.shape[0]}, batch={att_batch_size}, "
                    f"q_len={self.decode_max_q_seq_len}"
                )
            q_bshd = q.view(
                att_batch_size, self.decode_max_q_seq_len, q.shape[1], q.shape[2]
            )
            output = maca_flash_attn_with_kvcache(
                q=q_bshd,
                k_cache=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                v_cache=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
                block_table=self.page_table,
                cache_seqlens=self.b_att_seq_len,
                softmax_scale=sm_scale,
                causal=self.causal,
                window_size=window_size,
                softcap=0.0,
            )
            return output.view_as(q)
        elif q.device.type == "npu":
            import torch_npu

            N_Q = q.shape[-2]
            k, v, N_KV, kv_cache_args = self._prepare_npu_kv_cache(k, v)

            if self.decode_max_q_seq_len == 1 or self.use_mtp_bnsd:
                input_layout = "BNSD"
                sparse_mode = 0
                atten_mask = None
                # unsqueeze(2) on [B, H, D] yields non-contiguous [B, H, 1, D].
                # FIA graph_task_update then inserts AsStrided/aclnnContiguous
                # and CANN 8.5.1 fails with 207019.
                q = q.unsqueeze(2).contiguous()
            else:
                input_layout = "TND"
                sparse_mode = 3
                atten_mask = self.backend.get_causal_attn_mask(q.device)
                if not q.is_contiguous():
                    q = q.contiguous()
            if not k.is_contiguous():
                k = k.contiguous()
            if not v.is_contiguous():
                v = v.contiguous()
            page_table = self.page_table
            if page_table is not None and not page_table.is_contiguous():
                page_table = page_table.contiguous()
                self.page_table = page_table
            kv_cache_args = {
                name: val.contiguous() if isinstance(val, torch.Tensor) and not val.is_contiguous() else val
                for name, val in kv_cache_args.items()
            }

            output = torch.empty_like(q)
            softmax_lse = torch.empty(1, dtype=torch.float16, device=q.device)
            if torch.npu.is_current_stream_capturing():
                stream = torch.npu.current_stream()

                from lightllm.common.basemodel.graph.acl_graph import get_attn_params

                batch_size = self.infer_state.batch_size
                attn_params = get_attn_params()

                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)

                workspace = attn_params.workspaces.get(batch_size, None)
                if workspace is None:
                    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query=q,
                        key=k,
                        value=v,
                        atten_mask=atten_mask,
                        input_layout=input_layout,
                        sparse_mode=sparse_mode,
                        next_tokens=0,
                        scale=sm_scale,
                        actual_seq_lengths=self.infer_state.b1_cu_q_seq_len_cpu,
                        actual_seq_lengths_kv=self.infer_state.b_cu_kv_seq_len_cpu,
                        num_heads=N_Q,
                        num_key_value_heads=N_KV,
                        block_table=self.page_table,
                        block_size=self.backend.page_size,
                        **kv_cache_args,
                    )
                    attn_params.workspaces[batch_size] = workspace

                torch.npu.graph_task_group_begin(stream)
                torch_npu.npu_fused_infer_attention_score.out(
                    query=q,
                    key=k,
                    value=v,
                    atten_mask=atten_mask,
                    input_layout=input_layout,
                    sparse_mode=sparse_mode,
                    next_tokens=0,
                    scale=sm_scale,
                    actual_seq_lengths=self.infer_state.b1_cu_q_seq_len_cpu,
                    actual_seq_lengths_kv=self.infer_state.b_cu_kv_seq_len_cpu,
                    num_heads=N_Q,
                    num_key_value_heads=N_KV,
                    block_table=page_table,
                    block_size=self.backend.page_size,
                    workspace=workspace,
                    out=[output, softmax_lse],
                    **kv_cache_args,
                )
                handle = torch.npu.graph_task_group_end(stream)

                from lightllm.common.basemodel.graph.acl_graph import add_attn_params

                add_attn_params(
                    batch_size=self.infer_state.batch_size,
                    event=event,
                    handle=handle,
                    attn_params=(
                        weak_ref_tensor(q),
                        weak_ref_tensor(k),
                        weak_ref_tensor(v),
                        sm_scale,
                        N_Q,
                        N_KV,
                        weak_ref_tensor(page_table),
                        self.backend.page_size,
                        weak_ref_tensor(output),
                        weak_ref_tensor(softmax_lse),
                        weak_ref_tensor(atten_mask),
                        input_layout,
                        sparse_mode,
                        {name: weak_ref_tensor(value) for name, value in kv_cache_args.items()},
                    ),
                    microbatch_index=self.infer_state.microbatch_index,
                )
            else:
                torch_npu.npu_fused_infer_attention_score.out(
                    query=q,
                    key=k,
                    value=v,
                    atten_mask=atten_mask,
                    input_layout=input_layout,
                    sparse_mode=sparse_mode,
                    next_tokens=0,
                    scale=sm_scale,
                    actual_seq_lengths=self.infer_state.b1_cu_q_seq_len_cpu,
                    actual_seq_lengths_kv=self.infer_state.b_cu_kv_seq_len_cpu,
                    num_heads=N_Q,
                    num_key_value_heads=N_KV,
                    block_table=page_table,
                    block_size=self.backend.page_size,
                    out=[output, softmax_lse],
                    **kv_cache_args,
                )

            return output.squeeze(2) if input_layout == "BNSD" else output
        else:
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k.view(-1, self.backend.page_size, k.shape[1], k.shape[2]),
                v_cache=v.view(-1, self.backend.page_size, v.shape[1], v.shape[2]),
                page_table=self.page_table,
                cache_seqlens=self.b_att_seq_len,
                cu_seqlens_q=self.cu_seqlens_q,
                cu_seqlens_k_new=self.cu_seqlens_k,
                max_seqlen_q=self.decode_max_q_seq_len,
                softmax_scale=sm_scale,
                causal=self.causal,
                window_size=window_size,
                softcap=0.0,
                k_descale=None,
                v_descale=None,
                return_softmax_lse=False,
                sinks=sink_weight,
            )
