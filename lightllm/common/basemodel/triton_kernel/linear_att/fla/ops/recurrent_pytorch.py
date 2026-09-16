from __future__ import annotations

import torch

from .fused_recurrent import fused_recurrent_gated_delta_rule


@torch.no_grad()
def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    if head_first:
        raise NotImplementedError("head_first=True is not supported in the Ascend GDN fallback")

    assert q.dtype == k.dtype == v.dtype
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    assert HV % H == 0, f"HV ({HV}) must be divisible by H ({H})"

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("cu_seqlens requires batch size 1 (packed sequences)")
        cu = cu_seqlens.detach().to("cpu", dtype=torch.int64).tolist()
        N = len(cu) - 1
        if initial_state is not None and initial_state.shape[0] != N:
            raise ValueError(f"initial_state batch {initial_state.shape[0]} != num sequences {N}")
        seq_ranges = [(cu[n], cu[n + 1]) for n in range(N)]
    else:
        N = B
        seq_ranges = [(b * T, (b + 1) * T) for b in range(B)]
        # Equal-length path stores q as [B, T, ...]; flatten mentally via batch index.
        cu = None

    if initial_state is None:
        h_states = q.new_zeros(N, HV, K, V)
    else:
        h_states = initial_state.clone()

    o = v.new_empty(B, T if cu is not None else T, HV, V)

    if cu is None:
        # Equal-length: one fused call with B sequences works when cu_seqlens is None
        # and inplace_final_state=False (state written per token into [T,...] is wrong
        # for B>1). Process one batch element at a time.
        for b in range(B):
            o_b, ht = fused_recurrent_gated_delta_rule(
                q=q[b : b + 1],
                k=k[b : b + 1],
                v=v[b : b + 1],
                g=g[b : b + 1],
                beta=beta[b : b + 1],
                scale=scale,
                initial_state=h_states[b : b + 1],
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            # o_b: [1, T, HV, V] after API; ht: [T, HV, K, V]
            if o_b.dim() == 3:
                o[b] = o_b
            else:
                o[b] = o_b[0]
            h_states[b] = ht[T - 1]
    else:
        for n, (bos, eos) in enumerate(seq_ranges):
            if eos <= bos:
                continue
            o_n, ht = fused_recurrent_gated_delta_rule(
                q=q[:, bos:eos],
                k=k[:, bos:eos],
                v=v[:, bos:eos],
                g=g[:, bos:eos],
                beta=beta[:, bos:eos],
                scale=scale,
                initial_state=h_states[n : n + 1],
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            if o_n.dim() == 3:
                o[0, bos:eos] = o_n
            else:
                o[0, bos:eos] = o_n[0]
            h_states[n] = ht[eos - bos - 1]

    final_state = h_states if output_final_state else None
    return o, final_state
