import torch


def chunk_prefill_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if torch.npu.get_device_name(q.device) == "Ascend910B4":
        from .chunk_prefill_npu_910b4 import chunk_prefill_npu_910b4

        return chunk_prefill_npu_910b4(q, k, v, g, beta, initial_state, cu_seqlens, state_indices, scale)

    from .chunk_prefill_npu_910b2c import chunk_prefill_npu_910b2c

    return chunk_prefill_npu_910b2c(q, k, v, g, beta, initial_state, cu_seqlens, state_indices, scale)
