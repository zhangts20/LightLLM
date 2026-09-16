import torch
from typing import Optional

from .triton_impl import FuseMoeTriton


class FuseMoeNPUBase(FuseMoeTriton):

    def _select_experts(
        self,
        input_tensor: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: Optional[torch.Tensor],
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: int,
        num_expert_group: int,
        scoring_func: str,
        per_expert_scale: Optional[torch.Tensor] = None,
        shared_expert_gate: Optional[torch.Tensor] = None,
    ):
        if input_tensor.shape[0] != router_logits.shape[0]:
            raise ValueError("Input and router logits must contain the same number of tokens")

        router_logits = router_logits.float()
        if scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
        elif scoring_func == "softmax":
            scores = torch.softmax(router_logits, dim=-1)
        else:
            raise ValueError(f"Unsupported MoE scoring function: {scoring_func}")

        scores_for_choice = scores if correction_bias is None else scores + correction_bias.float()
        if use_grouped_topk:
            if topk_group is None or num_expert_group is None:
                raise ValueError("Grouped TopK requires topk_group and num_expert_group")
            if scores.shape[-1] % num_expert_group != 0:
                raise ValueError("The number of experts must be divisible by num_expert_group")

            grouped_scores = scores_for_choice.view(scores.shape[0], num_expert_group, -1)
            group_score_topk = 2 if (topk_group, num_expert_group, top_k) == (4, 8, 8) else 1
            group_scores = torch.topk(grouped_scores, k=group_score_topk, dim=-1).values.sum(dim=-1)
            selected_groups = torch.topk(group_scores, k=topk_group, dim=-1).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, selected_groups, True)
            expert_mask = group_mask.unsqueeze(-1).expand_as(grouped_scores).reshape_as(scores_for_choice)
            scores_for_choice = scores_for_choice.masked_fill(~expert_mask, float("-inf"))

        topk_ids = torch.topk(scores_for_choice, k=top_k, dim=-1).indices
        topk_weights = torch.gather(scores, 1, topk_ids)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        topk_weights = topk_weights.float()
        topk_ids = topk_ids.to(torch.int32)
        if self.routed_scaling_factor != 1.0:
            topk_weights.mul_(self.routed_scaling_factor)
        if per_expert_scale is not None:
            topk_weights.mul_(per_expert_scale[topk_ids.long()].float())

        routed_topk_ids = topk_ids
        if self.num_fused_shared_experts > 0:
            token_num = topk_ids.shape[0]
            shared_ids = (
                torch.arange(
                    self.n_routed_experts,
                    self.n_routed_experts + self.num_fused_shared_experts,
                    dtype=torch.int32,
                    device=topk_ids.device,
                )
                .view(1, -1)
                .expand(token_num, -1)
            )
            if shared_expert_gate is None:
                shared_weights = torch.ones(
                    (token_num, self.num_fused_shared_experts),
                    dtype=torch.float32,
                    device=topk_weights.device,
                )
            else:
                shared_expert_gate = shared_expert_gate.reshape(token_num, -1)
                if shared_expert_gate.shape[1] != self.num_fused_shared_experts:
                    raise ValueError("shared_expert_gate has an incompatible shape")
                shared_weights = torch.sigmoid(shared_expert_gate.float())
            topk_weights = torch.cat((topk_weights, shared_weights), dim=-1)
            topk_ids = torch.cat((topk_ids, shared_ids), dim=-1)

        return topk_weights, topk_ids, routed_topk_ids
