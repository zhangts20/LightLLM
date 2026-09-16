import torch


def gqa_token_decode_attention_flash_decoding(
    q: torch.Tensor,
    infer_state,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    out=None,
    alloc_tensor_func=torch.empty,
    sliding_window=(-1, -1),
):
    batch_size = infer_state.batch_size
    q_head_num, head_dim = q.shape[1], q.shape[2]
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    # Because we need to allocate some intermediate tensors, considering parallelism and
    # intermediate memory consumption, when batch_size is small, block_num is larger,
    # and when batch_size is large, block_num is smaller. This achieves a better balance
    # of memory consumption and performance.
    BLOCK_SEQ = 256
    if batch_size <= 16:
        block_num = 128
    elif batch_size <= 64:
        block_num = 64
    else:
        block_num = 32
    # Long KV: increase block_num so stage1 stays parallel instead of striding a small grid.
    max_kv_seq_len = int(getattr(infer_state, "max_kv_seq_len", 0) or 0)
    if max_kv_seq_len > 0:
        needed_blocks = (max_kv_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
        block_num = max(block_num, min(needed_blocks, 1024))

    mid_o = alloc_tensor_func([batch_size, q_head_num, block_num, head_dim], dtype=q.dtype, device=q.device)
    mid_o_logexpsum = alloc_tensor_func([batch_size, q_head_num, block_num], dtype=torch.float32, device=q.device)

    flash_decode_stage1(
        q=q.view(calcu_shape1),
        k=cache_k,
        v=cache_v,
        Req_to_tokens=infer_state.req_manager.req_to_token_indexs,
        B_req_idx=infer_state.b_req_idx,
        B_Seqlen=infer_state.b_seq_len,
        max_len_in_batch=infer_state.max_kv_seq_len,
        mid_out=mid_o,
        mid_out_logsumexp=mid_o_logexpsum,
        block_seq=BLOCK_SEQ,
        sliding_window=sliding_window,
    )
    flash_decode_stage2(
        mid_out=mid_o,
        mid_out_logexpsum=mid_o_logexpsum,
        B_Seqlen=infer_state.b_seq_len,
        out=o_tensor.view(calcu_shape1),
        block_seq=BLOCK_SEQ,
        sliding_window=sliding_window,
    )
    return o_tensor
