import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo


# @npu_timer
@torch.no_grad()
def npu_gqa_token_decode_attention_flash_decoding(
    q: torch.Tensor, infer_state: InferStateInfo, cache_k: torch.Tensor, cache_v: torch.Tensor, out=None, alloc_tensor_func=torch.empty
):
    import torch_npu

    # get kv
    Req_to_token_indexs = infer_state.req_manager.req_to_token_indexs
    req_indices = Req_to_token_indexs[infer_state.b_req_idx][:, :infer_state.max_kv_seq_len]
    k_padded = cache_k[req_indices]
    v_padded = cache_v[req_indices]

    q_shape = tuple(q.shape)
    q = q.view(q.shape[0], 1, q.shape[1], q.shape[2])
    scale_value = 1.0 / (q.shape[-1] ** 0.5)
    # https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_incre_flash_attention.md
    o = torch_npu.npu_incre_flash_attention(
        query=q,
        key=k_padded,
        value=v_padded,
        num_heads=q.shape[-2],
        num_key_value_heads=cache_k.shape[-2],
        scale_value=scale_value,
        input_layout="BSND",
        actual_seq_lengths=infer_state.b_kv_seq_len_cpu,
    )
    out.copy_(o.view(q_shape))


def gqa_token_decode_attention_flash_decoding(
    q: torch.Tensor, infer_state, cache_k: torch.Tensor, cache_v: torch.Tensor, out=None, alloc_tensor_func=torch.empty
):
    BLOCK_SEQ = 128
    batch_size = infer_state.batch_size
    max_kv_seq_len = infer_state.max_kv_seq_len
    q_head_num, head_dim = q.shape[1], q.shape[2]
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, q.dtype, q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1, head_dim], dtype=torch.float32, device=q.device
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_kv_seq_len // BLOCK_SEQ + 1], dtype=torch.float32, device=q.device
    )

    flash_decode_stage1(
        q.view(calcu_shape1),
        cache_k,
        cache_v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_kv_seq_len,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
