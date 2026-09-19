import dataclasses
from typing import Any

import torch

from lightllm.platform.base.attention import register_att_backend

from .fp import PagedFa3AttBackend, PagedFa3DecodeAttState, PagedFa3PrefillAttState
from .graph_utils import weak_ref_tensor
from ..base_att import AttControl


@register_att_backend(name="paged_fa3", category="standard", platforms=("ascend",), validate_name="fa3")
class PagedFa3AscendAttBackend(PagedFa3AttBackend):

    def get_causal_attn_mask(self, device):
        if not hasattr(self, "_causal_attn_mask"):
            self._causal_attn_mask = torch.triu(
                torch.ones((2048, 2048), dtype=torch.int8, device=device), diagonal=1
            )
        return self._causal_attn_mask

    def get_decode_seq_len_cpu_buffers(self, min_len: int):
        cap = max(min_len, self.model.graph_max_batch_size)
        if not hasattr(self, "_decode_seq_len_cpu_q") or self._decode_seq_len_cpu_q.shape[0] < min_len:
            self._decode_seq_len_cpu_q = torch.empty(cap, dtype=torch.int32, pin_memory=True)
            self._decode_seq_len_cpu_kv = torch.empty(cap, dtype=torch.int32, pin_memory=True)
        return self._decode_seq_len_cpu_q, self._decode_seq_len_cpu_kv

    def create_att_prefill_state(self, infer_state):
        return PagedFa3AscendPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return PagedFa3AscendDecodeAttState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class PagedFa3AscendPrefillAttState(PagedFa3PrefillAttState):

    def init_state(self):
        super().init_state()
        self.atten_mask = self.backend.get_causal_attn_mask(self.infer_state.input_ids.device)

    def _normal_prefill_att(self, q, k, v, att_control: AttControl, alloc_func=torch.empty):
        import torch_npu

        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
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


@dataclasses.dataclass
class PagedFa3AscendDecodeAttState(PagedFa3DecodeAttState):

    def init_state(self):
        args_mtp_step, _, rows, att_batch_size, last_rows, b_kv_seq_len = self._init_decode_layout()
        self.use_mtp_bnsd = args_mtp_step > 0 or not self.causal
        self._init_page_table_state(args_mtp_step, att_batch_size, rows, last_rows, b_kv_seq_len)
        if self.causal:
            return
        # Block draft is B=width Q_S=1 IncreFA. Graph replay reads these
        # buffers, so they must be ones(B) + per-token KV.
        q_cu = torch.zeros(rows + 1, dtype=torch.int32)
        q_cu[1:] = 1
        kv_seqlens = torch.tensor(self._npu_block_kv_seqlens(rows), dtype=torch.int32)
        self.infer_state.seq_len_manager.update(q_cu, kv_seqlens)
        (
            self.infer_state.b1_cu_q_seq_len_cpu,
            self.infer_state.b_cu_kv_seq_len_cpu,
        ) = self.infer_state.seq_len_manager.get_tensor_slices()

    def _prepare_npu_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
        N_KV, HEAD_DIM = k.shape[-2:]
        k = k.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
        v = v.view(-1, self.backend.page_size, N_KV * HEAD_DIM)
        return k, v, N_KV, {}

    def _npu_decode_query_layout(self, q: torch.Tensor):
        if self.decode_max_q_seq_len == 1 or self.use_mtp_bnsd:
            # unsqueeze(2) on [B, H, D] yields non-contiguous [B, H, 1, D].
            # FIA graph_task_update then inserts AsStrided/aclnnContiguous.
            q = q.unsqueeze(2).contiguous()
            return q, "BNSD", 0, None, "incre"
        atten_mask = self.backend.get_causal_attn_mask(q.device)
        if not q.is_contiguous():
            q = q.contiguous()
        return q, "TND", 3, atten_mask, "tnd"

    def _npu_decode_output(self, output: torch.Tensor, q_kind: str) -> torch.Tensor:
        if q_kind == "incre":
            return output.squeeze(2)
        return output

    def _npu_fia_kv_seqlens(self, kv_cpu: torch.Tensor) -> torch.Tensor:
        if not torch.npu.is_current_stream_capturing():
            return kv_cpu
        max_kv = int(self.backend.model.graph_max_len_in_batch)
        q_cpu, kv_buf = self.backend.get_decode_seq_len_cpu_buffers(kv_cpu.numel())
        del q_cpu
        n = kv_cpu.numel()
        kv_buf[:n].fill_(max_kv)
        return kv_buf[:n]

    def _npu_fia_kwargs(
        self,
        q,
        k,
        v,
        *,
        sm_scale,
        N_Q,
        N_KV,
        input_layout,
        sparse_mode,
        atten_mask,
        q_seqlens,
        kv_seqlens,
        page_table=None,
        block_size=0,
        kv_cache_args=None,
    ):
        kwargs = {
            "query": q,
            "key": k,
            "value": v,
            "atten_mask": atten_mask,
            "input_layout": input_layout,
            "sparse_mode": sparse_mode,
            "next_tokens": 0,
            "scale": sm_scale,
            "actual_seq_lengths": q_seqlens,
            "actual_seq_lengths_kv": kv_seqlens,
            "num_heads": N_Q,
            "num_key_value_heads": N_KV,
        }
        if page_table is not None:
            kwargs["block_table"] = page_table
            kwargs["block_size"] = block_size
        if kv_cache_args:
            kwargs.update(kv_cache_args)
        return kwargs

    def _npu_run_fia(
        self,
        q,
        k,
        v,
        *,
        sm_scale,
        N_Q,
        N_KV,
        input_layout,
        sparse_mode,
        atten_mask,
        q_seqlens,
        kv_seqlens,
        page_table=None,
        block_size=0,
        kv_cache_args=None,
        q_kind="tnd",
    ):
        import torch_npu

        kv_cache_args = kv_cache_args or {}
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        if page_table is not None and not page_table.is_contiguous():
            page_table = page_table.contiguous()
            self.page_table = page_table
        kv_cache_args = {
            name: val.contiguous() if isinstance(val, torch.Tensor) and not val.is_contiguous() else val
            for name, val in kv_cache_args.items()
        }

        output = torch.empty_like(q)
        softmax_lse = torch.empty(1, dtype=torch.float16, device=q.device)
        run_kwargs = self._npu_fia_kwargs(
            q,
            k,
            v,
            sm_scale=sm_scale,
            N_Q=N_Q,
            N_KV=N_KV,
            input_layout=input_layout,
            sparse_mode=sparse_mode,
            atten_mask=atten_mask,
            q_seqlens=q_seqlens,
            kv_seqlens=kv_seqlens,
            page_table=page_table,
            block_size=block_size,
            kv_cache_args=kv_cache_args,
        )
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
                workspace_kwargs = dict(run_kwargs)
                workspace_kwargs["actual_seq_lengths_kv"] = self._npu_fia_kv_seqlens(kv_seqlens)
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**workspace_kwargs)
                attn_params.workspaces[batch_size] = workspace

            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                **run_kwargs,
                workspace=workspace,
                out=[output, softmax_lse],
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
                    block_size,
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
                **run_kwargs,
                out=[output, softmax_lse],
            )

        return self._npu_decode_output(output, q_kind)

    def _npu_block_kv_seqlens(self, q_tokens: int) -> list[int]:
        lens = self.b_att_seq_len.detach().cpu().tolist()
        if len(lens) == q_tokens:
            return [int(x) for x in lens]
        width = max(int(self.decode_max_q_seq_len), 1)
        out: list[int] = []
        for i, kv_len in enumerate(lens):
            n_q = width if i + 1 < len(lens) else q_tokens - i * width
            out.extend([int(kv_len)] * max(n_q, 0))
        if len(out) < q_tokens:
            out.extend([int(lens[-1])] * (q_tokens - len(out)))
        return out[:q_tokens]

    def _normal_decode_att(self, q, k, v, att_control: AttControl, alloc_func=torch.empty):
        if q.shape[0] != self.infer_state.batch_size:
            raise ValueError("Unexpected NPU decode query shape")
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        N_Q = q.shape[-2]
        k, v, N_KV, kv_cache_args = self._prepare_npu_kv_cache(k, v)
        q, input_layout, sparse_mode, atten_mask, q_kind = self._npu_decode_query_layout(q)
        return self._npu_run_fia(
            q,
            k,
            v,
            sm_scale=sm_scale,
            N_Q=N_Q,
            N_KV=N_KV,
            input_layout=input_layout,
            sparse_mode=sparse_mode,
            atten_mask=atten_mask,
            q_seqlens=self.infer_state.b1_cu_q_seq_len_cpu,
            kv_seqlens=self._npu_fia_kv_seqlens(self.infer_state.b_cu_kv_seq_len_cpu),
            page_table=self.page_table,
            block_size=self.backend.page_size,
            kv_cache_args=kv_cache_args,
            q_kind=q_kind,
        )
