import dataclasses

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


# Ascend FIA lives in fp_npu.py (PagedFa3AscendAttBackend).
@register_att_backend(name="paged_fa3", category="standard", platforms=("cuda", "maca"), validate_name="fa3")
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

    def _init_decode_layout(self):
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

        mtp_size = args_mtp_step + 1
        rows = self.infer_state.batch_size
        self.has_partial_block = rows % mtp_size != 0
        if not is_block_mode:
            assert not self.has_partial_block
        att_batch_size = triton.cdiv(rows, mtp_size)
        last_rows = None
        b_kv_seq_len = None
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
        return args_mtp_step, mtp_size, rows, att_batch_size, last_rows, b_kv_seq_len

    def _init_page_table_state(self, args_mtp_step, att_batch_size, rows, last_rows, b_kv_seq_len):
        model = self.backend.model
        page_table_batch_size = rows if self.use_mtp_bnsd else att_batch_size
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

    def _init_noncausal_block_state(self, args_mtp_step, mtp_size, rows, last_rows, b_kv_seq_len):
        self.b_block_req_idx = (
            self.infer_state.b_req_idx.index_select(0, last_rows)
            if args_mtp_step > 0 else self.infer_state.b_req_idx
        )
        self.b_att_seq_len = b_kv_seq_len.contiguous() if args_mtp_step > 0 else self.infer_state.b_seq_len
        self.decode_max_q_seq_len = mtp_size
        self.block_max_kv_len = self.infer_state.max_kv_seq_len
        # CUDA/MetaX capture a static token table / packed gather of graph_max_len.
        model = self.backend.model
        if rows <= model.graph_max_batch_size and self.block_max_kv_len <= model.graph_max_len_in_batch:
            self.block_max_kv_len = model.graph_max_len_in_batch

    def init_state(self):
        args_mtp_step, mtp_size, rows, att_batch_size, last_rows, b_kv_seq_len = self._init_decode_layout()
        if not self.causal and (
            type(self)._normal_decode_att is not PagedFa3DecodeAttState._normal_decode_att
            and not getattr(self, "supports_noncausal_block", False)
        ):
            raise NotImplementedError(
                "paged_fa3 noncausal block decode requires a supported CUDA or MACA backend"
            )
        self.use_mtp_bnsd = False
        if not self.causal:
            self._init_noncausal_block_state(args_mtp_step, mtp_size, rows, last_rows, b_kv_seq_len)
            return
        self._init_page_table_state(args_mtp_step, att_batch_size, rows, last_rows, b_kv_seq_len)

    def decode_att(self, q, k, v, att_control: AttControl = AttControl(), alloc_func=torch.empty):
        assert att_control.use_alibi is False
        return self._normal_decode_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

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
                raise ValueError("Unexpected GPU decode query shape for noncausal block")
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
