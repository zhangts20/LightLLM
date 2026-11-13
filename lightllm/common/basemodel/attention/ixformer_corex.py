import torch
import triton

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.triton_kernel.fa3_utils import paged_page_table_copy_kernel
from lightllm.utils.envs_utils import get_env_start_args, get_page_size


class CorexBlockTable:

    def __init__(self, device):
        self.page_size = get_page_size()

        start_args = get_env_start_args()
        self.rows, self.pages = self.get_buffer_shape(start_args)

        if start_args.enable_decode_microbatch_overlap or start_args.enable_prefill_microbatch_overlap:
            self.slots = 2
        else:
            self.slots = 1
        self.buffer = torch.empty(self.slots * self.rows * self.pages, dtype=torch.int32, device=device)

    def get_buffer_shape(self, start_args) -> tuple[int, int]:
        req_rows = max(start_args.graph_max_batch_size, start_args.running_max_req_size)
        verify_width = 1
        if start_args.mtp_mode is not None and start_args.mtp_step > 0:
            verify_width = start_args.mtp_step + 1
        rows = req_rows * verify_width
        seq = max(start_args.graph_max_len_in_batch, start_args.max_req_total_len)
        if rows <= 0 or seq <= 0:
            raise ValueError(
                "CoreX block table shape is invalid: "
                f"batch rows={rows} (max of graph_max_batch_size and running_max_req_size, "
                f"times MTP verify width {verify_width}), "
                f"seq len={seq} (max of graph_max_len_in_batch and max_req_total_len). "
                "Both must be > 0."
            )
        return rows, (seq + self.page_size - 1) // self.page_size

    def get_page_table(
        self,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        max_kv_seq_len: int,
        slot: int = 0,
        *,
        view_len: int,
    ) -> torch.Tensor:
        view_pages = (view_len + self.page_size - 1) // self.page_size
        fill_pages = (max_kv_seq_len + self.page_size - 1) // self.page_size
        batch = b_req_idx.shape[0]
        start = slot * self.rows * self.pages
        need = batch * view_pages
        slot_size = self.rows * self.pages
        if need > slot_size:
            raise RuntimeError(
                "CoreX page table is "
                f"{self.rows} rows x {self.pages} pages, "
                f"but this step needs batch {batch} x {view_pages} pages"
            )
        page_table = self.buffer[start : start + need].view(batch, view_pages)
        self._fill_page_table(fill_pages, page_table, req_to_token_indexs, b_req_idx)
        return page_table

    def get_view_len(self, max_kv_seq_len: int, graph=None, batch_size: int = 0) -> int:
        if graph is not None and graph.can_run(batch_size, max_kv_seq_len):
            return graph.graph_max_len_in_batch
        return max_kv_seq_len

    def _fill_page_table(
        self,
        num_pages: int,
        page_table: torch.Tensor,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
    ) -> None:
        BLOCK_SIZE = 128
        grid = (page_table.shape[0], triton.cdiv(num_pages, BLOCK_SIZE))
        paged_page_table_copy_kernel[grid](
            page_table,
            req_to_token_indexs,
            b_req_idx,
            num_pages,
            b_req_idx.stride(0),
            page_table.stride(0),
            page_table.stride(1),
            req_to_token_indexs.stride(0),
            req_to_token_indexs.stride(1),
            PAGE_SIZE=self.page_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )


def _window(att_control: AttControl) -> tuple[int, int]:
    if att_control.use_sliding_window:
        left, right = att_control.sliding_window
        return left, right
    return (-1, -1)


def _use_decode_kvcache(q: torch.Tensor, max_kv_seq_len: int, max_query_len: int) -> bool:
    if max_query_len > 1:
        return True
    if q.shape[-1] >= 512:
        return False
    if torch.cuda.is_current_stream_capturing():
        return True
    return q.shape[0] >= 4 or max_kv_seq_len >= 1024


def ixformer_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    causal: bool,
    att_control: AttControl,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor | None = None,
    max_query_len: int = 1,
    cu_query_lens: torch.Tensor | None = None,
) -> None:
    scale = q.shape[-1] ** -0.5
    if cache_seqlens is not None and _use_decode_kvcache(q, max_kv_seq_len, max_query_len):
        from ixformer.contrib.vllm_flash_attn import flash_attn_with_kvcache

        if cu_query_lens is None:
            cu_query_lens = cu_seqlens_q

        flash_attn_with_kvcache(
            q.unsqueeze(1),
            k,
            v,
            block_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=scale,
            causal=causal,
            window_size=_window(att_control),
            out=out.unsqueeze(1),
            max_query_len=max_query_len,
            cu_query_lens=cu_query_lens.to(dtype=torch.int32),
            max_context_len=max_kv_seq_len,
            use_cuda_graph=torch.cuda.is_current_stream_capturing(),
        )
        return

    from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func

    flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q.to(dtype=torch.int32),
        cu_seqlens_k.to(dtype=torch.int32),
        max_q_seq_len,
        max_kv_seq_len,
        causal=causal,
        window_size=_window(att_control),
        softmax_scale=scale,
        block_table=page_table,
        out=out,
    )
