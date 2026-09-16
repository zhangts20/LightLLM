import torch

_QUERY_PARTITION_SIZE = 4096


def can_use_prefix_flash(q: torch.Tensor, k: torch.Tensor) -> bool:
    return (
        q.dtype in (torch.bfloat16, torch.float16)
        and q.shape[2] == 256
        and 4096 <= q.shape[0] <= 32768
        and 4096 <= k.shape[0] <= 131072
    )


def prefix_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape[0] >= 8192 and k.shape[0] >= 16384:
        parts = [_prefix_flash_attention(q_part, k, v) for q_part in q.split(_QUERY_PARTITION_SIZE)]
        return torch.cat([part[0] for part in parts], dim=0), torch.cat([part[1] for part in parts], dim=0)
    return _prefix_flash_attention(q, k, v)


def _prefix_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    tokens, heads, dim = q.shape
    output, softmax_max, softmax_sum, *_ = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        heads,
        "TND",
        scale=dim**-0.5,
        keep_prob=1.0,
        sparse_mode=0,
        actual_seq_qlen=[tokens],
        actual_seq_kvlen=[k.shape[0]],
    )
    logsumexp = softmax_max[..., :1] + torch.log(softmax_sum[..., :1])
    logsumexp = logsumexp.reshape(heads, tokens, 1).transpose(0, 1)
    return output, logsumexp
