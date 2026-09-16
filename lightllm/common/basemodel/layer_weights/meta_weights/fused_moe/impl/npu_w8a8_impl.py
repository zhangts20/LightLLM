import torch
from typing import Optional

from lightllm.common.quantization.quantize_method import WeightPack

from .npu_base_impl import FuseMoeNPUBase


ACL_FORMAT_FRACTAL_NZ = 29


class FuseMoeNPUW8A8(FuseMoeNPUBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._npu_w8a8_weight_checked = False

    def _check_inputs(self, input_tensor: torch.Tensor, w13: WeightPack, w2: WeightPack) -> None:
        import torch_npu

        if input_tensor.device.type != "npu":
            raise RuntimeError("FuseMoeNPUW8A8 requires NPU input tensors")

        if input_tensor.dtype != torch.bfloat16:
            raise NotImplementedError("FuseMoeNPUW8A8 currently supports only bfloat16 hidden states")

        if input_tensor.ndim != 2:
            raise ValueError(f"FuseMoeNPUW8A8 expects a 2D input, but got shape {tuple(input_tensor.shape)}")

        if w13.weight.ndim != 3 or w2.weight.ndim != 3:
            raise ValueError("FuseMoeNPUW8A8 expects 3D expert weights")

        if w13.weight.dtype != torch.int8 or w2.weight.dtype != torch.int8:
            raise TypeError("FuseMoeNPUW8A8 expects int8 expert weights")

        if w13.weight.shape[0] != w2.weight.shape[0]:
            raise ValueError("w13 and w2 must contain the same number of experts")

        expert_num, hidden_size, fused_intermediate_size = w13.weight.shape
        intermediate_size = fused_intermediate_size // 2
        if fused_intermediate_size % 2 != 0:
            raise ValueError("The last dimension of w13 must be divisible by 2 for SwiGLU")

        if w2.weight.shape != (expert_num, intermediate_size, hidden_size):
            raise ValueError(
                f"Incompatible W8A8 MoE weight shapes: w13={tuple(w13.weight.shape)}, w2={tuple(w2.weight.shape)}"
            )

        if input_tensor.shape[1] != hidden_size:
            raise ValueError(
                f"Input hidden size {input_tensor.shape[1]} does not match weight hidden size {hidden_size}"
            )

        if expert_num > 256:
            raise NotImplementedError("Ascend grouped-matmul finalize-routing supports at most 256 experts")

        if hidden_size < 256 or hidden_size % 32 != 0:
            raise ValueError("The output hidden size must be at least 256 and divisible by 32")

        if intermediate_size % 16 != 0:
            raise ValueError("The MoE intermediate size must be divisible by 16")

        expected_w13_scale_shape = (expert_num, fused_intermediate_size)
        expected_w2_scale_shape = (expert_num, hidden_size)

        if w13.weight_scale is None or tuple(w13.weight_scale.shape) != expected_w13_scale_shape:
            raise ValueError(f"w13 weight scale must have shape {expected_w13_scale_shape}")

        if w2.weight_scale is None or tuple(w2.weight_scale.shape) != expected_w2_scale_shape:
            raise ValueError(f"w2 weight scale must have shape {expected_w2_scale_shape}")

        if w13.weight_scale.dtype != torch.float32 or w2.weight_scale.dtype != torch.float32:
            raise TypeError("FuseMoeNPUW8A8 expects float32 per-channel weight scales")

        if not self._npu_w8a8_weight_checked:
            if torch_npu.get_npu_format(w13.weight) != ACL_FORMAT_FRACTAL_NZ:
                raise RuntimeError("w13 must be converted to FRACTAL_NZ before W8A8 MoE execution")
            if torch_npu.get_npu_format(w2.weight) != ACL_FORMAT_FRACTAL_NZ:
                raise RuntimeError("w2 must be converted to FRACTAL_NZ before W8A8 MoE execution")
            self._npu_w8a8_weight_checked = True

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
        topk_weights = topk_weights.to(dtype=torch.float32).contiguous()
        routed_input = input_tensor if input_tensor.is_contiguous() else input_tensor.contiguous()
        expert_num = w13.weight.shape[0]
        expanded_token_num = token_num * topk_num

        routed_input_q, expanded_row_idx, expert_token_count, routed_input_scale = torch_npu.npu_moe_init_routing_v2(
            routed_input,
            topk_ids,
            active_num=expanded_token_num,
            expert_capacity=-1,
            expert_num=expert_num,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            quant_mode=1,
            active_expert_range=[0, expert_num],
            # Return expert-sorted row -> original flattened TopK row.
            row_idx_type=1,
        )

        # GroupedMatmulSwiGluQuant uses cumulative expert-token counts.
        expert_token_cumsum = torch.cumsum(expert_token_count, dim=0)
        intermediate_q, intermediate_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
            routed_input_q,
            w13.weight,
            expert_token_cumsum,
            w13.weight_scale,
            routed_input_scale,
        )

        expanded_row_idx_long = expanded_row_idx.to(dtype=torch.int64)
        sorted_topk_weights = torch.index_select(topk_weights.reshape(-1), 0, expanded_row_idx_long)
        output_row_idx = torch.div(expanded_row_idx, topk_num, rounding_mode="floor")

        # The current operator requires shared_input when logit and output_bs
        # are provided. A zero coefficient lets us reuse the input tensor
        # without contributing it to the MoE result.
        output = torch_npu.npu_grouped_matmul_finalize_routing(
            intermediate_q,
            w2.weight,
            expert_token_count,
            scale=w2.weight_scale,
            pertoken_scale=intermediate_scale,
            shared_input=routed_input,
            logit=sorted_topk_weights,
            row_index=output_row_idx,
            dtype=torch.float32,
            shared_input_weight=0.0,
            output_bs=token_num,
            group_list_type=1,
        )

        # Existing LightLLM TP-MoE callers rely on in-place output.
        input_tensor.copy_(output)
        return input_tensor
