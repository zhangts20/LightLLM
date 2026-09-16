from typing import Optional, Union

import torch

try:
    from flashinfer.decode import fast_decode_plan as _FAST_DECODE_PLAN
except ImportError:
    _FAST_DECODE_PLAN = None


def use_maca_cuda_graph_uniform_kv_layout(model, infer_state) -> bool:
    if model.platform_backend.name != "maca":
        return False

    graph = getattr(model, "graph", None)
    if graph is None:
        return False

    return graph.can_run(infer_state.batch_size, infer_state.max_kv_seq_len)


def should_init_decode_wrapper(model, infer_state) -> bool:
    graph = getattr(model, "graph", None)
    if graph is None:
        # Cuda graph is disabled, so this state owns a normal decode wrapper.
        return True

    if infer_state.is_cuda_graph:
        # This is the captured graph state; it must create the wrapper captured by replay.
        return True

    if not graph.can_run(infer_state.batch_size, infer_state.max_kv_seq_len):
        # Cuda graph is enabled, but this input falls outside graph limits and runs normally.
        return True

    # This is a temporary replay state. Its tensors are copied into the captured graph state.
    return False


def refresh_cuda_graph_decode_plan(
    decode_wrapper,
    *,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int = 1,
    q_data_type: Optional[Union[str, torch.dtype]] = None,
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = True,
    global_override_indptr_cpu: Optional[torch.Tensor] = None,
) -> None:
    if _FAST_DECODE_PLAN is not None:
        _FAST_DECODE_PLAN(
            decode_wrapper,
            indptr=indptr,
            indices=indices,
            last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            non_blocking=non_blocking,
            global_override_indptr_cpu=global_override_indptr_cpu,
        )
        return

    _refresh_cuda_graph_decode_plan(
        decode_wrapper,
        batch_size=len(last_page_len),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        indptr_host=global_override_indptr_cpu,
    )


def _refresh_cuda_graph_decode_plan(
    decode_wrapper,
    *,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    indptr_host: Optional[torch.Tensor],
) -> None:
    import flashinfer.decode as flashinfer_decode

    cached_module = getattr(decode_wrapper, "_cached_module", None)
    assert cached_module is not None and indptr_host is not None, (
        "CUDA Graph decode replay requires a captured FlashInfer module and a host padded kv indptr"
    )
    assert decode_wrapper.use_tensor_cores and page_size == 1, (
        "CUDA Graph decode plan refresh expects use_tensor_cores=True and page_size=1"
    )

    qo_indptr_host = flashinfer_decode._get_range_buf(batch_size + 1, "cpu")
    kv_lens_arr_host = (indptr_host[1:] - indptr_host[:-1]).contiguous()
    with decode_wrapper.device as device:
        decode_wrapper._plan_info = cached_module.plan(
            decode_wrapper._float_workspace_buffer,
            decode_wrapper._int_workspace_buffer,
            decode_wrapper._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            indptr_host,
            kv_lens_arr_host,
            batch_size,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            decode_wrapper.is_cuda_graph_enabled,
            head_dim,
            head_dim,
            False,
            flashinfer_decode.get_cuda_stream(device),
        )
