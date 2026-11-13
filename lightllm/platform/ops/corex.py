import torch
from typing import Any, Callable, Optional, Tuple

import ixformer.inference.functions as ixf

from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import gated_rmsnorm_forward
from lightllm.common.basemodel.triton_kernel.norm.layernorm import layernorm_forward
from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_fused_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.platform.base.ops import out_like, register_op

_BF16_DTYPES = (torch.float16, torch.bfloat16)


@register_op("corex")
def ffn(
    *,
    input: torch.Tensor,
    layer_weight: Any,
    alloc_func: Callable,
    embed_dim: int,
) -> torch.Tensor:
    input = input.view(-1, embed_dim)
    up_gate_out = layer_weight.gate_up_proj.mm(input)
    ffn1_out = alloc_func(
        (input.size(0), up_gate_out.size(1) // 2),
        dtype=input.dtype,
        device=input.device,
    )
    if input.dtype in _BF16_DTYPES:
        ixf.silu_and_mul(up_gate_out, ffn1_out)
    else:
        silu_and_mul_fwd(up_gate_out, ffn1_out)
    return layer_weight.down_proj.mm(ffn1_out)


@register_op(
    "corex",
    out=lambda kwargs: (
        (kwargs["weight"].shape[0], kwargs["input"].shape[1]),
        kwargs["weight"].dtype,
        kwargs["weight"].device,
    ),
)
def lm_head(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    # MR-V100 上 M=1、K>=4096 的 torch.mm 只有约 150 GB/s。ixformer linear 约 0.9 ms，
    # 和 torch.mm 的差在 bf16 舍入内。ixf.mm 在这个 K 上别用。
    if (
        input.shape[1] == 1
        and input.shape[0] >= 4096
        and input.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
    ):
        ixf.linear(input.reshape(1, -1), weight, output=out.view(1, -1))
        return out
    torch.mm(weight, input, out=out)
    return out


@register_op("corex", out=out_like("input"))
def rms_norm(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    gate_value: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # GDN 的 norm(x) * silu(z) 不是 ixformer.rms_norm。
    if gate_value is not None:
        gated_rmsnorm_forward(x=input, weight=weight, bias=None, eps=eps, z=gate_value, out=out)
    elif input.dtype in _BF16_DTYPES:
        ixf.rms_norm(input, weight, eps=eps, output=out)
    else:
        from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward

        rmsnorm_forward(x=input, weight=weight, eps=eps, out=out)
    return out


@register_op("corex")
def qk_rms_norm(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    eps: float,
    fp32_multiply: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fp32_multiply or q.dtype not in _BF16_DTYPES:
        return qk_rmsnorm_fused_forward(q=q, k=k, w_q=w_q, w_k=w_k, eps=eps, fp32_multiply=fp32_multiply)
    # ixformer 要求最后一维是 head_dim。Qwen 传入的是 (tokens, heads * head_dim)，
    # K 还经常是 KV 拼接张量的前缀切片，view 成 (tokens, heads, head_dim) 后原地写回。
    q_heads = q if q.ndim == 3 else q.view(q.shape[0], -1, w_q.shape[0])
    k_heads = k if k.ndim == 3 else k.view(k.shape[0], -1, w_k.shape[0])
    ixf.rms_norm_qk(
        input_q=q_heads,
        input_k=k_heads,
        weight_q=w_q,
        weight_k=w_k,
        eps=eps,
        output_q=q_heads,
        output_k=k_heads,
    )
    return q, k


@register_op("corex")
def rotary_emb(
    *,
    is_prefill: bool,
    batch_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_rotary_factor: float = 1.0,
    rotary_impl: Optional[Callable] = None,
) -> None:
    del is_prefill, batch_size
    # ixformer apply_rope 要先拼出完整 rotary 维，Qwen3.6 只转前 64 维，实测比 Triton 慢一个数量级。
    impl = rotary_impl or rotary_emb_fwd
    impl(q=q, k=k, cos=cos, sin=sin, partial_rotary_factor=partial_rotary_factor)


def residual_rms_norm(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
) -> torch.Tensor:
    """Fused ``rms_norm(hidden + delta)``.

    Not part of ``OpsProtocol``, so it is not registered as a dispatchable op --
    callers import it directly (see ``Qwen3NextTransformerLayerInfer``).
    """
    ixf.residual_rms_norm(
        input,
        weight,
        eps=eps,
        residual=residual,
        output=out,
        residual_output=residual_out,
    )
    return out


@register_op("corex", out=out_like("input"))
def layer_norm(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    if input.dtype in _BF16_DTYPES and weight.dtype == input.dtype and bias.dtype == input.dtype:
        ixf.layer_norm(input, [input.shape[-1]], weight, bias, eps=eps, output=out)
        return out
    out.copy_(layernorm_forward(x=input, weight=weight, bias=bias, eps=eps))
    return out
