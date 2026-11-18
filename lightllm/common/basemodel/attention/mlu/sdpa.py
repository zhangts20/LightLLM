"""MLU prefill attention using the native SDPA implementation."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend, TritonPrefillAttState
from lightllm.platform.base.attention import register_att_backend


@dataclass
class PrefillPlan:
    query_start: int
    query_length: int
    prefix_length: int
    kv_indices: torch.Tensor | None
    kv_slice: tuple[int, int] | None
    padded_query: torch.Tensor | None = None


def build_prefill_plan(req_indices, query_starts, sequence_lengths, prefix_lengths, req_to_tokens):
    count = sequence_lengths.numel()
    if not count:
        return []
    packed = torch.cat([req_indices, query_starts, sequence_lengths, prefix_lengths]).cpu().tolist()
    maximum = max(packed[2 * count : 3 * count])
    rows = req_to_tokens.index_select(0, req_indices.long())[:, :maximum].cpu().tolist()
    plans = []
    for index in range(count):
        full_length = packed[2 * count + index]
        prefix = packed[3 * count + index]
        if not 0 <= prefix <= full_length:
            raise ValueError("Cached prefix must be within the sequence")
        if full_length == prefix:
            continue
        locations = rows[index][:full_length]
        contiguous = all(value == locations[0] + offset for offset, value in enumerate(locations))
        plans.append(
            PrefillPlan(
                packed[count + index],
                full_length - prefix,
                prefix,
                None if contiguous else torch.tensor(locations, device=req_to_tokens.device, dtype=torch.int64),
                (locations[0], locations[0] + full_length) if contiguous else None,
            )
        )
    return plans


def paged_prefill_sdpa(query, key, value, output, plans):
    """Apply global causal positions, including cached prefixes and paged KV."""
    query_heads, kv_heads = query.shape[1], key.shape[1]
    if query_heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count")
    for plan in plans:
        if plan.kv_slice is not None:
            lo, hi = plan.kv_slice
            current_key, current_value = key[lo:hi], value[lo:hi]
        else:
            current_key = key.index_select(0, plan.kv_indices)
            current_value = value.index_select(0, plan.kv_indices)
        current_key = current_key.transpose(0, 1).unsqueeze(0)
        current_value = current_value.transpose(0, 1).unsqueeze(0)
        if query_heads != kv_heads:
            groups = query_heads // kv_heads
            current_key = current_key.repeat_interleave(groups, dim=1)
            current_value = current_value.repeat_interleave(groups, dim=1)
        start, length, prefix = plan.query_start, plan.query_length, plan.prefix_length
        current_query = query[start : start + length].transpose(0, 1).unsqueeze(0)
        if prefix:
            shape = (1, query_heads, prefix + length, query.shape[2])
            if plan.padded_query is None or plan.padded_query.shape != shape:
                plan.padded_query = query.new_zeros(shape)
            plan.padded_query[:, :, prefix:].copy_(current_query)
            current_query = plan.padded_query
        result = F.scaled_dot_product_attention(current_query, current_key, current_value, is_causal=True)
        output[start : start + length].copy_(result[0, :, prefix:].transpose(0, 1))


@register_att_backend(name="sdpa", category="standard", platforms=("mlu",), validate_name="triton")
class MluSdpaAttBackend(TritonAttBackend):
    def create_att_prefill_state(self, infer_state):
        return MluSdpaPrefillState(backend=self, infer_state=infer_state)


@dataclass
class MluSdpaPrefillState(TritonPrefillAttState):
    def _nomarl_prefill_att(self, q, k, v, att_control, alloc_func=torch.empty):
        # Host metadata reads cannot be captured. Keep graph and windowed paths
        # on the existing Triton implementation.
        if self.infer_state.is_cuda_graph or att_control.use_sliding_window:
            return super()._nomarl_prefill_att(q, k, v, att_control, alloc_func)
        state = self.infer_state
        plans = getattr(state, "_mlu_sdpa_prefill_plan", None)
        if plans is None:
            plans = build_prefill_plan(
                state.b_req_idx,
                state.b_q_start_loc,
                state.b_seq_len,
                state.b_ready_cache_len,
                state.req_manager.req_to_token_indexs,
            )
            state._mlu_sdpa_prefill_plan = plans
        output = alloc_func(q.shape, q.dtype, device=q.device)
        paged_prefill_sdpa(q, k, v, output, plans)
        return output
