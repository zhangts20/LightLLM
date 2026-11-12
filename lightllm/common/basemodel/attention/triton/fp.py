import dataclasses
import torch
import triton
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl
from lightllm.platform.base.attention import register_att_backend
from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers


@register_att_backend(name="triton", category="standard", platforms=("cuda", "musa", "maca", "mlu"))
class TritonAttBackend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state) -> "TritonPrefillAttState":
        return TritonPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "TritonDecodeAttState":
        return TritonDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class TritonPrefillAttState(BasePrefillAttState):
    def init_state(self):
        pass

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        if att_control.use_alibi:
            assert att_control.use_sliding_window is False, "alibi + sliding_window not supported"
            assert att_control.tp_alibi is not None
            return self._alibi_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)
        else:
            return self._nomarl_prefill_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)

    def _alibi_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        out = alloc_func(q.shape, q.dtype)

        from ...triton_kernel.alibi_att.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q,
            k,
            v,
            out,
            self.infer_state.b_req_idx,
            att_control.tp_alibi,
            self.infer_state.b_q_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.max_q_seq_len,
            self.infer_state.req_manager.req_to_token_indexs,
        )
        return out

    def _nomarl_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.att.prefill_att.context_flashattention_nopad import context_attention_fwd

        if att_control.use_sliding_window:
            sliding_window = att_control.sliding_window
        else:
            sliding_window = (-1, -1)

        out = alloc_func(q.shape, q.dtype, device=q.device)
        context_attention_fwd(
            q,
            k,
            v,
            out,
            self.infer_state.b_req_idx,
            self.infer_state.b_q_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.b_ready_cache_len,
            self.infer_state.max_q_seq_len,
            self.infer_state.req_manager.req_to_token_indexs,
            sliding_window=sliding_window,
        )
        return out


@dataclasses.dataclass
class TritonDecodeAttState(BaseDecodeAttState):
    b_mark_mtp_shared_group: torch.Tensor = None

    def init_state(self):
        draft_step = self.backend.model.mtp_manager.get_decode_draft_step(self.backend.model.is_mtp_draft_model)
        if draft_step > 0:
            self.b_mark_mtp_shared_group = build_mtp_shared_group_markers(
                self.infer_state.b_req_idx,
                hold_req_id=self.backend.model.req_manager.HOLD_REQUEST_ID,
            )

    def copy_for_decode_cuda_graph(self, new_state: "TritonDecodeAttState"):
        super().copy_for_decode_cuda_graph(new_state)

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        if att_control.use_alibi:
            assert att_control.use_sliding_window is False, "alibi + sliding_window not supported"
            assert att_control.tp_alibi is not None
            return self._alibi_decode_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)
        else:
            draft_step = self.backend.model.mtp_manager.get_decode_draft_step(self.backend.model.is_mtp_draft_model)

            q_head_num = q.shape[1]
            k_head_num = k.shape[1]

            if draft_step > 0:
                assert q_head_num >= k_head_num, "speculative decode requires q_head_num >= k_head_num"
                return self._spec_decode_gqa_att(q=q, k=k, v=v, alloc_func=alloc_func)
            elif q_head_num == k_head_num:
                assert att_control.use_sliding_window is False, "sliding_window not supported in non-gqa attention yet"
                return self._normal_decode_flash_decoding_att(q=q, k=k, v=v, alloc_func=alloc_func)
            elif q_head_num > k_head_num:
                return self._normal_decode_gqa_flash_decoding_att(
                    q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func
                )
            else:
                raise NotImplementedError("error")

    def _alibi_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.alibi_att.token_flashattention_nopad import token_attention_fwd

        out = alloc_func(q.shape, q.dtype)
        token_attention_fwd(
            q,
            k,
            v,
            out,
            att_control.tp_alibi,
            self.infer_state.req_manager.req_to_token_indexs,
            self.infer_state.b_req_idx,
            self.infer_state.b_kv_start_loc,
            self.infer_state.b_seq_len,
            self.infer_state.max_kv_seq_len,
            self.infer_state.total_token_num,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _normal_decode_flash_decoding_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.att.decode_att.mha.flash_decoding.flash_decoding import (
            token_decode_attention_flash_decoding,
        )

        out = alloc_func(q.shape, q.dtype, device=q.device)

        token_decode_attention_flash_decoding(
            q=q,
            infer_state=self.infer_state,
            cache_k=k,
            cache_v=v,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _normal_decode_gqa_flash_decoding_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding import (
            gqa_token_decode_attention_flash_decoding,
        )

        if att_control.use_sliding_window:
            sliding_window = att_control.sliding_window
        else:
            sliding_window = (-1, -1)

        out = alloc_func(q.shape, q.dtype, device=q.device)

        gqa_token_decode_attention_flash_decoding(
            q=q,
            infer_state=self.infer_state,
            cache_k=k,
            cache_v=v,
            out=out,
            alloc_tensor_func=alloc_func,
            sliding_window=sliding_window,
        )

        return out

    def _spec_decode_gqa_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        from ...triton_kernel.att.decode_att.gqa.mtp_diverse import (
            token_decode_attention_mtp_diverse_single_token,
        )

        out = token_decode_attention_mtp_diverse_single_token(
            q=q,
            k=k,
            v=v,
            Req_to_tokens=self.infer_state.req_manager.req_to_token_indexs,
            B_req_idx=self.infer_state.b_req_idx,
            b_seq_len=self.infer_state.b_seq_len,
            b_mark_shared_group=self.b_mark_mtp_shared_group,
            alloc_tensor_func=alloc_func,
        )

        return out

    def _normal_decode_gqa_flash_decoding_att_vsm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        # TODO USE , 在特定场景下比 _normal_decode_gqa_flash_decoding_att 省显存
        from ...triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_vsm import (
            gqa_token_decode_attention_flash_decoding_vsm,
        )

        out = alloc_func(q.shape, q.dtype)

        gqa_token_decode_attention_flash_decoding_vsm(
            q=q,
            k=k,
            v=v,
            infer_state=self.infer_state,
            out=out,
            alloc_tensor_func=alloc_func,
        )
        return out

    def _normal_decode_gqa_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_weight,
        alloc_func=torch.empty,
    ):
        # TODO USE , 在特定场景下比 _normal_decode_gqa_flash_decoding_att 省显存
        from ...triton_kernel.att.decode_att.gqa.gqa_decode_flashattention_nopad import gqa_decode_attention_fwd

        out = alloc_func(q.shape, q.dtype)

        gqa_decode_attention_fwd(
            q=q,
            k=k,
            v=v,
            out=out,
            req_to_tokens=self.infer_state.req_manager.req_to_token_indexs,
            b_req_idx=self.infer_state.b_req_idx,
            b_seq_len=self.infer_state.b_seq_len,
        )
        return out

    def _normal_decode_stage3_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alloc_func=torch.empty,
    ):
        total_token_num = self.infer_state.total_token_num
        batch_size = self.infer_state.batch_size
        q_head_num = q.shape[1]
        head_dim = q.shape[2]

        calcu_shape1 = (batch_size, q_head_num, head_dim)
        att_m_tensor = alloc_func((q_head_num, total_token_num), torch.float32)

        from ...triton_kernel.att.decode_att.mha.stage3_decode_att.token_attention_nopad_att1 import token_att_fwd

        token_att_fwd(
            q.view(calcu_shape1),
            k,
            att_m_tensor,
            Req_to_tokens=self.infer_state.req_manager.req_to_token_indexs,
            B_req_idx=self.infer_state.b_req_idx,
            B_Start_Loc=self.infer_state.b_kv_start_loc,
            B_Seqlen=self.infer_state.b_seq_len,
            max_len_in_batch=self.infer_state.max_kv_seq_len,
        )

        o_tensor = alloc_func(q.shape, q.dtype)
        from ...triton_kernel.att.decode_att.mha.stage3_decode_att.token_attention_softmax_and_reducev import (
            token_softmax_reducev_fwd,
        )

        token_softmax_reducev_fwd(
            att_m_tensor,
            v,
            o_tensor.view(calcu_shape1),
            req_to_tokens=self.infer_state.req_manager.req_to_token_indexs,
            b_req_idx=self.infer_state.b_req_idx,
            b_start_loc=self.infer_state.b_kv_start_loc,
            b_seq_len=self.infer_state.b_seq_len,
        )
        return o_tensor


def _fill_corex_page_table(
    backend,
    infer_state,
    for_decode_graph: bool,
    b_req_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, int]:
    block_table = getattr(infer_state.mem_manager, "block_table", None)
    if block_table is None:
        return None, infer_state.max_kv_seq_len

    graph = getattr(backend.model, "graph", None) if for_decode_graph else None
    view_len = block_table.get_view_len(
        infer_state.max_kv_seq_len,
        graph=graph,
        batch_size=infer_state.batch_size,
    )
    page_table = block_table.get_page_table(
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx if b_req_idx is None else b_req_idx,
        infer_state.max_kv_seq_len,
        slot=infer_state.microbatch_index,
        view_len=view_len,
    )
    return page_table, view_len


def _corex_query_group(att_backend) -> int:
    width = att_backend.model.mtp_manager.get_decode_batch_multiplier(
        att_backend.model.is_mtp_draft_model
    )
    if width <= 1:
        return 1
    if att_backend.uses_dynamic_spec_verify_layout():
        raise NotImplementedError("CoreX paged KV requires mtp_dynamic_verify=False")
    return width


def _ixformer_attention(
    infer_state,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    att_control: AttControl,
    page_table: torch.Tensor,
    *,
    causal: bool,
    max_kv_seq_len: int,
    cache_seqlens: torch.Tensor | None = None,
    max_query_len: int = 1,
    cu_query_lens: torch.Tensor | None = None,
) -> None:
    from ..ixformer_corex import ixformer_varlen_attention

    ixformer_varlen_attention(
        q,
        k,
        v,
        out,
        cu_seqlens_q=infer_state.b1_cu_q_seq_len,
        cu_seqlens_k=infer_state.b1_cu_kv_seq_len,
        max_q_seq_len=max(max_query_len, infer_state.max_q_seq_len),
        max_kv_seq_len=max_kv_seq_len,
        causal=causal,
        att_control=att_control,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_query_len=max_query_len,
        cu_query_lens=cu_query_lens,
    )


@register_att_backend(name="triton", category="standard", platforms=("corex"))
class CoreXTritonAttBackend(TritonAttBackend):

    def create_att_prefill_state(self, infer_state) -> "CoreXTritonPrefillAttState":
        return CoreXTritonPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state) -> "CoreXTritonDecodeAttState":
        return CoreXTritonDecodeAttState(backend=self, infer_state=infer_state)


class CoreXTritonPrefillAttState(TritonPrefillAttState):
    page_table: torch.Tensor = None
    page_kv_view_len: int = 0

    def init_state(self):
        self.page_table, self.page_kv_view_len = _fill_corex_page_table(
            self.backend, self.infer_state, for_decode_graph=False
        )

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        if self.page_table is None:
            return super().prefill_att(q, k, v, att_control=att_control, alloc_func=alloc_func)
        return self._nomarl_prefill_att(q, k, v, att_control=att_control, alloc_func=alloc_func)

    def _nomarl_prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        if self.page_table is None:
            return super()._nomarl_prefill_att(q, k, v, att_control=att_control, alloc_func=alloc_func)
        if att_control.use_alibi:
            raise NotImplementedError("CoreX paged KV does not support alibi attention")
        out = alloc_func(q.shape, q.dtype, device=q.device)
        _ixformer_attention(
            self.infer_state,
            q,
            k,
            v,
            out,
            att_control,
            self.page_table,
            causal=self.backend.uses_causal_attention(),
            max_kv_seq_len=self.page_kv_view_len,
        )
        return out


class CoreXTritonDecodeAttState(TritonDecodeAttState):
    page_table: torch.Tensor = None
    page_kv_view_len: int = 0
    query_width: int = 1
    b_group_seq_len: torch.Tensor = None
    b1_cu_group_q_lens: torch.Tensor = None

    def init_state(self):
        super().init_state()
        self.query_width = _corex_query_group(self.backend)
        if self.query_width == 1:
            self.page_table, self.page_kv_view_len = _fill_corex_page_table(
                self.backend, self.infer_state, for_decode_graph=True
            )
            return

        groups = triton.cdiv(self.infer_state.batch_size, self.query_width)
        last_rows = torch.arange(
            self.query_width - 1, self.infer_state.batch_size, self.query_width,
            dtype=torch.int64,
            device=self.infer_state.b_seq_len.device,
        )
        group_seq_len = self.infer_state.b_seq_len.index_select(0, last_rows).int()
        self.b_group_seq_len = group_seq_len.clamp_min(self.query_width).contiguous()
        self.b1_cu_group_q_lens = torch.arange(
            0, groups + 1, 
            dtype=torch.int32,
            device=self.infer_state.b_seq_len.device,
        ) * self.query_width
        self.page_table, self.page_kv_view_len = _fill_corex_page_table(
            self.backend,
            self.infer_state,
            for_decode_graph=True,
            b_req_idx=self.infer_state.b_req_idx.index_select(0, last_rows),
        )

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        if self.page_table is None:
            return super().decode_att(q, k, v, att_control=att_control, alloc_func=alloc_func)
        if att_control.use_alibi:
            raise NotImplementedError("CoreX paged KV does not support alibi attention")
        if self.query_width > 1:
            return self._grouped_decode_att(q=q, k=k, v=v, att_control=att_control, alloc_func=alloc_func)
        return self._single_row_decode_att(q, k, v, att_control, alloc_func)

    def _grouped_decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl,
        alloc_func,
    ) -> torch.Tensor:
        out = alloc_func(q.shape, q.dtype, device=q.device)
        _ixformer_attention(
            self.infer_state,
            q,
            k,
            v,
            out,
            att_control,
            self.page_table,
            causal=True,
            max_kv_seq_len=self.page_kv_view_len,
            cache_seqlens=self.b_group_seq_len,
            max_query_len=self.query_width,
            cu_query_lens=self.b1_cu_group_q_lens,
        )
        return out

    def _single_row_decode_att(self, q, k, v, att_control, alloc_func) -> torch.Tensor:
        out = alloc_func(q.shape, q.dtype, device=q.device)
        _ixformer_attention(
            self.infer_state,
            q,
            k,
            v,
            out,
            att_control,
            self.page_table,
            causal=True,
            max_kv_seq_len=self.page_kv_view_len,
            cache_seqlens=self.infer_state.b_seq_len,
        )
        return out
