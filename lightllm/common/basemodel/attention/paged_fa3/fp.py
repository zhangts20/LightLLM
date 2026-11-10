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
    import flash_attn_2_cuda as maca_flash_attn_cuda
except ImportError:
    maca_flash_attn_varlen_func = None
    maca_flash_attn_with_kvcache = None
    maca_flash_attn_cuda = None


@triton.jit(do_not_specialize=["softmax_scale"])
def _causal_tail_merge_kernel(
    Q,
    Kcache,
    Vcache,
    Out1,
    Lse1,
    Out,
    BlockTable,
    SeqLens,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kp,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vp,
    stride_vh,
    stride_vd,
    stride_o1b,
    stride_o1h,
    stride_o1d,
    stride_lb,
    stride_lh,
    stride_ob,
    stride_oh,
    stride_od,
    stride_tb,
    stride_tp,
    n_q,
    group,
    page_size,
    softmax_scale,
    DIM: tl.constexpr,
    N_TAIL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < DIM
    q = tl.load(Q + batch * stride_qb + head * stride_qh + offs * stride_qd, mask=mask, other=0.0).to(tl.float32)
    seqlen = tl.load(SeqLens + batch)
    query_i = head // n_q
    kv_head = (head % n_q) // group

    neg = float("-inf")
    max_score = neg
    exp_sum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for t in range(N_TAIL):
        pos = seqlen - N_TAIL + t
        keep = (pos >= 0) & (t < query_i)
        pos_c = tl.maximum(pos, 0)
        page = pos_c // page_size
        offset = pos_c % page_size
        block = tl.load(BlockTable + batch * stride_tb + page * stride_tp)
        k = tl.load(
            Kcache + block * stride_kb + offset * stride_kp + kv_head * stride_kh + offs * stride_kd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            Vcache + block * stride_vb + offset * stride_vp + kv_head * stride_vh + offs * stride_vd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(q * k, axis=0) * softmax_scale
        new_max = tl.where(keep, tl.maximum(max_score, score), max_score)
        alpha = tl.where(keep, tl.exp(max_score - new_max), 1.0)
        prob = tl.where(keep, tl.exp(score - new_max), 0.0)
        acc = acc * alpha + prob * v
        exp_sum = exp_sum * alpha + prob
        max_score = new_max

    has_tail = exp_sum > 0
    out2 = acc / tl.where(has_tail, exp_sum, 1.0)
    lse2 = tl.where(has_tail, max_score + tl.log(tl.where(has_tail, exp_sum, 1.0)), neg)

    has_prefix = seqlen > N_TAIL
    lse1 = tl.where(has_prefix, tl.load(Lse1 + batch * stride_lb + head * stride_lh), neg)
    out1 = tl.load(
        Out1 + batch * stride_o1b + head * stride_o1h + offs * stride_o1d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out1 = tl.where(has_prefix, out1, 0.0)

    has_any = has_prefix | has_tail
    peak = tl.maximum(tl.where(has_prefix, lse1, neg), lse2)
    peak = tl.where(has_any, peak, 0.0)
    lse = peak + tl.log(tl.exp(tl.where(has_prefix, lse1, neg) - peak) + tl.exp(lse2 - peak))
    merged = tl.where(has_prefix, tl.exp(lse1 - lse), 0.0) * out1 + tl.where(has_tail, tl.exp(lse2 - lse), 0.0) * out2
    tl.store(
        Out + batch * stride_ob + head * stride_oh + offs * stride_od,
        merged.to(Out.dtype.element_ty),
        mask=mask,
    )


def _merge_causal_tail(q_flat, k_cache, v_cache, out1, lse1, block_table, cache_seqlens, softmax_scale, n_q, n_tail):
    batch, q_heads, dim = q_flat.shape
    lse = lse1.float()
    if lse.shape[-1] == 1:
        lse = lse.squeeze(-1)
    else:
        lse = torch.logsumexp(lse, dim=-1)
    lse = lse.contiguous()
    out = torch.empty_like(q_flat)
    block_d = 1 << (dim - 1).bit_length()
    _causal_tail_merge_kernel[(batch, q_heads)](
        q_flat,
        k_cache,
        v_cache,
        out1,
        lse,
        out,
        block_table,
        cache_seqlens,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        out1.stride(0),
        out1.stride(1),
        out1.stride(2),
        lse.stride(0),
        lse.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        n_q,
        n_q // k_cache.shape[2],
        k_cache.shape[1],
        softmax_scale,
        DIM=dim,
        N_TAIL=n_tail,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def _maca_decode_num_splits(batch):
    return 16 if batch >= 64 else 0


def maca_grouped_causal_kvcache(q_bshd, k_cache, v_cache, block_table, cache_seqlens, softmax_scale):
    batch, q_len, n_q, dim = q_bshd.shape
    qq = q_bshd.reshape(batch, 1, q_len * n_q, dim)
    prefix = (cache_seqlens - (q_len - 1)).clamp_min(0)
    out1, lse1 = maca_flash_attn_cuda.fwd_kvcache(
        qq,
        k_cache,
        v_cache,
        None,
        None,
        prefix,
        None,
        None,
        None,
        None,
        block_table,
        None,
        None,
        softmax_scale,
        False,
        -1,
        -1,
        0.0,
        True,
        _maca_decode_num_splits(batch),
    )
    merged = _merge_causal_tail(
        qq[:, 0],
        k_cache,
        v_cache,
        out1[:, 0],
        lse1,
        block_table,
        cache_seqlens,
        softmax_scale,
        n_q,
        q_len - 1,
    )
    return merged.view(batch, q_len, n_q, dim)


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

    def _maca_kvcache_decode(self, q_bshd, k_cache, v_cache, sm_scale, window_size, alloc_func):
        q_len = q_bshd.shape[1]
        group = q_bshd.shape[2] // k_cache.shape[2]
        if q_len > 1 and group > q_len:
            if window_size == (-1, -1) and maca_flash_attn_cuda is not None:
                return maca_grouped_causal_kvcache(
                    q_bshd,
                    k_cache,
                    v_cache,
                    self.page_table,
                    self.b_att_seq_len,
                    sm_scale,
                )
            out = alloc_func(q_bshd.shape, dtype=q_bshd.dtype, device=q_bshd.device)
            for i in range(q_len):
                seqlens_i = (self.b_att_seq_len - (q_len - 1 - i)).clamp_min(0)
                out[:, i : i + 1] = maca_flash_attn_with_kvcache(
                    q=q_bshd[:, i : i + 1],
                    k_cache=k_cache,
                    v_cache=v_cache,
                    block_table=self.page_table,
                    cache_seqlens=seqlens_i,
                    softmax_scale=sm_scale,
                    causal=self.causal,
                    window_size=window_size,
                    softcap=0.0,
                )
            return out
        return maca_flash_attn_with_kvcache(
            q=q_bshd,
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=self.page_table,
            cache_seqlens=self.b_att_seq_len,
            softmax_scale=sm_scale,
            causal=self.causal,
            window_size=window_size,
            softcap=0.0,
            num_splits=_maca_decode_num_splits(q_bshd.shape[0]),
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
            k_cache = k.view(-1, self.backend.page_size, k.shape[1], k.shape[2])
            v_cache = v.view(-1, self.backend.page_size, v.shape[1], v.shape[2])
            output = self._maca_kvcache_decode(
                q_bshd, k_cache, v_cache, sm_scale, window_size, alloc_func
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
