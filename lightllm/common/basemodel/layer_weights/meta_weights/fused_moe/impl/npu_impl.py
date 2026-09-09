import torch
from typing import Optional

from lightllm.common.quantization.quantize_method import WeightPack

from .npu_base_impl import FuseMoeNPUBase


class FuseMoeNPU(FuseMoeNPUBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._npu_weight_checked = False
        self._w13_gmm_weight = None
        self._w2_gmm_weight = None

    def _check_inputs(self, input_tensor: torch.Tensor, w13: WeightPack, w2: WeightPack) -> None:
        if input_tensor.device.type != "npu":
            raise RuntimeError("FuseMoeNPU requires NPU input tensors")

        if input_tensor.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError("FuseMoeNPU supports float16 and bfloat16 hidden states")

        if input_tensor.ndim != 2:
            raise ValueError(f"FuseMoeNPU expects a 2D input, but got shape {tuple(input_tensor.shape)}")

        if w13.weight.ndim != 3 or w2.weight.ndim != 3:
            raise ValueError("FuseMoeNPU expects 3D expert weights")

        if w13.weight.dtype != input_tensor.dtype or w2.weight.dtype != input_tensor.dtype:
            raise TypeError("Floating-point MoE weights and hidden states must have the same dtype")

        expert_num, fused_intermediate_size, hidden_size = w13.weight.shape
        if fused_intermediate_size % 2 != 0:
            raise ValueError("The second dimension of w13 must be divisible by 2 for SwiGLU")

        intermediate_size = fused_intermediate_size // 2
        if w2.weight.shape != (expert_num, hidden_size, intermediate_size):
            raise ValueError(
                "Incompatible floating-point MoE weight shapes: "
                f"w13={tuple(w13.weight.shape)}, w2={tuple(w2.weight.shape)}"
            )

        if input_tensor.shape[1] != hidden_size:
            raise ValueError(
                f"Input hidden size {input_tensor.shape[1]} does not match weight hidden size {hidden_size}"
            )

        if not self._npu_weight_checked:
            # NoQuant stores linear weights as [E, N, K], while Ascend GMM
            # consumes [E, K, N]. These are metadata-only transpose views.
            self._w13_gmm_weight = w13.weight.transpose(-1, -2)
            self._w2_gmm_weight = w2.weight.transpose(-1, -2)
            self._npu_weight_checked = True

    def _fused_experts(
        self,
        input_tensor: torch.Tensor,
        w13: WeightPack,
        w2: WeightPack,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        del router_logits, is_prefill

        import torch_npu

        self._check_inputs(input_tensor, w13, w2)

        token_num = input_tensor.shape[0]
        topk_num = topk_ids.shape[1]
        if token_num == 0 or topk_num == 0:
            return input_tensor

        if topk_ids.shape != topk_weights.shape:
            raise ValueError("topk_ids and topk_weights must have the same shape")

        topk_ids = topk_ids.to(dtype=torch.int32).contiguous()
        topk_weights = topk_weights.to(dtype=input_tensor.dtype).contiguous()
        routed_input = input_tensor if input_tensor.is_contiguous() else input_tensor.contiguous()
        expert_num = w13.weight.shape[0]
        expanded_token_num = token_num * topk_num

        expanded_input, expanded_row_idx, expert_token_count, _ = torch_npu.npu_moe_init_routing_v2(
            routed_input,
            topk_ids,
            active_num=expanded_token_num,
            expert_capacity=-1,
            expert_num=expert_num,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=-1,
            active_expert_range=[0, expert_num],
            # npu_moe_finalize_routing consumes original-row -> sorted-row.
            row_idx_type=0,
        )
        intermediate = torch_npu.npu_grouped_matmul(
            x=[expanded_input],
            weight=[self._w13_gmm_weight],
            group_list=expert_token_count,
            split_item=2,
            group_type=0,
            group_list_type=1,
            output_dtype=input_tensor.dtype,
        )[0]
        intermediate = torch_npu.npu_swiglu(intermediate)
        expert_output = torch_npu.npu_grouped_matmul(
            x=[intermediate],
            weight=[self._w2_gmm_weight],
            group_list=expert_token_count,
            split_item=2,
            group_type=0,
            group_list_type=1,
            output_dtype=input_tensor.dtype,
        )[0]

        # V2 returns the inverse permutation in token-major TopK order, while
        # finalize-routing expects the legacy TopK-major order.
        finalize_row_idx = expanded_row_idx.view(token_num, topk_num).transpose(0, 1).contiguous().view(-1)
        output = torch_npu.npu_moe_finalize_routing(
            expert_output,
            None,
            None,
            None,
            topk_weights,
            finalize_row_idx,
            topk_ids,
            0,
        )

        input_tensor.copy_(output)
        return input_tensor
