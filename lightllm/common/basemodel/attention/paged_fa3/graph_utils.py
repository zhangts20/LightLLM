import torch
from typing import Any, Sequence, Tuple


def sync_attn_params(
    *,
    batch_size: int,
    seqlens_by_microbatch_index: Sequence[Tuple[Any, Any]],
    update_stream: Any,
    attn_params: Any,
) -> None:
    if batch_size == 0:
        return
    update_attn_params(batch_size, seqlens_by_microbatch_index, update_stream, attn_params)


def weak_ref_tensor(tensor: Any) -> Any:
    import torch_npu

    if isinstance(tensor, torch.Tensor):
        return torch_npu._C._weak_ref_tensor(tensor)
    return tensor


def update_attn_params(
    batch_size: int,
    seqlens_by_microbatch_index: Sequence[Tuple[Any, Any]],
    update_stream: Any,
    attn_params: Any,
):
    import torch_npu
    workspace = attn_params.workspaces[batch_size]

    with torch.npu.stream(update_stream):
        for microbatch_index, (actual_seq_lengths, actual_seq_lengths_kv) in enumerate(seqlens_by_microbatch_index):
            handles = attn_params.handles[batch_size][microbatch_index]
            events = attn_params.events[batch_size][microbatch_index]
            params_list = attn_params.attn_params[batch_size][microbatch_index]
            for handle, event, attn_param in zip(handles, events, params_list):
                (
                    q,
                    k,
                    v,
                    sm_scale,
                    N_Q,
                    N_KV,
                    page_table,
                    block_size,
                    output,
                    softmax_lse,
                    atten_mask,
                    input_layout,
                    sparse_mode,
                    kv_cache_args,
                ) = attn_param
                torch.npu.graph_task_update_begin(update_stream, handle)
                torch_npu.npu_fused_infer_attention_score.out(
                    q,
                    k,
                    v,
                    atten_mask=atten_mask,
                    input_layout=input_layout,
                    sparse_mode=sparse_mode,
                    next_tokens=0,
                    scale=sm_scale,
                    actual_seq_lengths=actual_seq_lengths,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_heads=N_Q,
                    num_key_value_heads=N_KV,
                    block_table=page_table,
                    block_size=block_size,
                    workspace=workspace,
                    out=[output, softmax_lse],
                    **kv_cache_args,
                )
                torch.npu.graph_task_update_end(update_stream)
                event.record(update_stream)
