from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, List, Tuple

import torch

from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    linear_att_mtp_state_index_update,
    mtp_scatter_next_token_ids,
    mtp_verify,
)

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
        MtpMemIndexesToFree,
        SpecProposal,
    )


def alloc_mem_indexes(token_count: int, *, allocations: List[MtpMemIndexesToFree]) -> torch.Tensor:
    """Return exact-size scratch indices and record the full allocation for cleanup."""

    token_count = int(token_count)
    if token_count == 0:
        return torch.empty((0,), dtype=torch.int32, device="cpu")

    from lightllm.server.router.model_infer.infer_batch import g_infer_context

    req_manager = g_infer_context.req_manager
    alloc_count = req_manager.get_page_aligned_mem_size(token_count)
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(alloc_count)
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import MtpMemIndexesToFree

    full_indexes = req_manager.alloc_page_aligned_mem_indices(token_count)
    allocations.append(MtpMemIndexesToFree(mem_indexes_cpu=full_indexes))
    return full_indexes[:token_count]


def alloc_eagle_mem_indexes(
    b_seq_len: torch.Tensor,
    b_last_mem_index: torch.Tensor,
    recursive_steps: int,
    *,
    allocations: List[MtpMemIndexesToFree],
) -> torch.Tensor:
    """Allocate step-major EAGLE rows at their logical paged KV offsets.

    b_seq_len and b_last_mem_index describe accepted target tails. Reuse the
    unused suffix of their pages in the separate draft KV layers, then allocate
    fresh pages at logical boundaries. Only those fresh pages belong to this
    proposal; the accepted target page must never enter scratch cleanup.
    """
    req_num = b_seq_len.numel()
    if req_num == 0 or recursive_steps == 0:
        return torch.empty((0,), dtype=torch.int32, device="cpu")

    from lightllm.server.router.model_infer.infer_batch import g_infer_context
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import MtpMemIndexesToFree

    req_manager = g_infer_context.req_manager
    if req_manager.get_page_aligned_mem_size(1) == 1:
        # Unpaged slots have no logical-offset constraints. Allocate the whole
        # proposal without synchronizing either accepted-tail tensor to CPU.
        return alloc_mem_indexes(req_num * recursive_steps, allocations=allocations)

    seq_len_cpu = b_seq_len.to(device="cpu")
    last_index_cpu = b_last_mem_index.to(device="cpu")
    assert seq_len_cpu.shape == last_index_cpu.shape
    step_indexes = []
    for step in range(1, recursive_steps + 1):
        next_seq_len = seq_len_cpu + step
        if g_infer_context.radix_cache is not None:
            need = req_manager.calc_real_need_token_num(req_num, next_seq_len)
            g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(need)
        last_index_cpu = req_manager.alloc_mem_indices(
            req_num, b_seq_len=next_seq_len, b_last_mem_index=last_index_cpu
        )
        step_indexes.append(last_index_cpu)
    indexes = torch.cat(step_indexes)
    owned_pages = req_manager.get_mtp_rejected_mem_indices_to_free(indexes)
    if owned_pages.numel():
        allocations.append(MtpMemIndexesToFree(mem_indexes_cpu=owned_pages))
    return indexes


def verify_mtp_tokens(
    backend: ModeBackend,
    next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    b_mtp_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verify target tokens and update recurrent MTP state when required."""

    accept_lengths, accepted_index = mtp_verify(
        req_to_next_token_ids=backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=next_token_ids,
        b_req_idx=b_req_idx,
    )
    if backend.is_linear_att_mixed_model:
        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=backend.model.req_manager.req_to_mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=backend.max_draft_step + 1,
        )
    return accept_lengths, accepted_index


def scatter_mtp_next_tokens(
    backend: ModeBackend,
    proposal: SpecProposal,  # proposal.token_ids: [req_num, draft_step]
    target_next_token_ids: torch.Tensor,  # [verify_batch_size]
    b_req_mtp_start_loc: torch.Tensor,  # [req_num]
    b_req_idx: torch.Tensor,  # [verify_batch_size]
    mtp_accept_len: torch.Tensor,  # [req_num]
) -> None:
    """Persist the next MTP proposal and optional scheduling scores by request."""

    schedule_scores = getattr(proposal, "schedule_scores", None)
    if schedule_scores is not None and 0 in schedule_scores.shape:
        schedule_scores = None

    sampling_params_manager = backend.model.req_manager.req_sampling_params_manager
    mtp_scatter_next_token_ids(
        req_to_next_token_ids=sampling_params_manager.req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        target_next_token_ids=target_next_token_ids,
        draft_token_ids=proposal.token_ids,
        b_req_idx=b_req_idx,
        mtp_accept_len=mtp_accept_len,
        req_to_next_token_scores=(
            sampling_params_manager.req_to_next_token_scores if schedule_scores is not None else None
        ),
        schedule_scores=schedule_scores,
    )


def record_request_mtp_metrics(
    backend: ModeBackend,
    decode_reqs: List[InferReq],
    accept_lengths_cpu: torch.Tensor,
    verify_run_reqs: List[InferReq],
) -> None:
    """Accumulate user-visible MTP metrics on each request."""

    if not backend.is_master_in_dp:
        return

    accept_lengths = accept_lengths_cpu.tolist()
    assert len(accept_lengths) == len(decode_reqs)
    verify_count_by_req_idx = Counter(req.req_idx for req in verify_run_reqs)
    for req, accept_len in zip(decode_reqs, accept_lengths):
        req.update_mtp_accepted_token_num(accept_token_num=accept_len - 1)
        verify_token_num = verify_count_by_req_idx[req.req_idx]
        if verify_token_num > 0:
            req.update_mtp_verify_token_num(verify_token_num=verify_token_num)
            req.update_mtp_verify_step_num(verify_step_num=1)


def free_mem_indexes(
    backend: ModeBackend,
    extra_mem_indexes_cpu: List[MtpMemIndexesToFree],
) -> None:
    """Free all KV indexes described by the unified MTP memory list."""

    mem_indexes_to_free = []
    for extra_mem_to_free in extra_mem_indexes_cpu:
        extra_indexes_cpu = extra_mem_to_free.mem_indexes_cpu
        if extra_mem_to_free.free_mask_cpu is not None:
            extra_indexes_cpu = backend.model.req_manager.get_mtp_rejected_mem_indices_to_free(
                extra_indexes_cpu[extra_mem_to_free.free_mask_cpu]
            )
        if extra_indexes_cpu.numel() > 0:
            mem_indexes_to_free.append(extra_indexes_cpu)

    if mem_indexes_to_free:
        backend.model.req_manager.mem_manager.free(torch.cat(mem_indexes_to_free, dim=0))


__all__ = [
    "alloc_mem_indexes",
    "alloc_eagle_mem_indexes",
    "free_mem_indexes",
    "record_request_mtp_metrics",
    "scatter_mtp_next_tokens",
    "verify_mtp_tokens",
]


def compact_decode_input(backend, model_input, req_num, plan):
    """Compact verify rows without changing ownership of allocated KV pages."""
    from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager

    model_input.to_device(backend.backend_runtime.target_device())
    width = backend.max_draft_step + 1
    assert model_input.batch_size == req_num * width
    request_ids = model_input.b_req_idx[::width].long()
    scores = backend.model.req_manager.req_sampling_params_manager.req_to_next_token_scores
    # Work on a copy: proposal scores remain conditional probabilities.
    scores = scores.index_select(0, request_ids)[:, : plan.pre_draft_step + 1].float().clone()
    scores.clamp_(min=0.01, max=0.99)
    scores[:, 0] = 1.0
    scores = scores.cumprod(dim=1)
    selected = torch.topk(scores.flatten(), k=plan.dynamic_batch_size, sorted=False).indices
    rows = (selected // scores.shape[1] * width + selected % scores.shape[1]).sort().values
    mask = torch.zeros(model_input.batch_size, dtype=torch.bool, device=request_ids.device)
    mask[rows] = True
    selected_cpu = g_pin_mem_manager.async_copy_from_gpu_tensor_with_event(
        key="selected_row_mask", gpu_tensor=mask
    )
    selected_cpu.wait()
    cpu_mask = selected_cpu.tensor
    old_indexes = model_input.mem_indexes_cpu
    to_free = backend.model.req_manager.get_mtp_rejected_mem_indices_to_free(old_indexes[~cpu_mask])
    if to_free.numel():
        backend.model.req_manager.mem_manager.free(to_free)
    model_input.mem_indexes_cpu = old_indexes[cpu_mask]
    for name in (
        "input_ids", "b_req_idx", "b_mtp_index", "b_seq_len", "b_position_delta",
        "b_shared_seq_len", "b_shared_radix_node_id", "mem_indexes", "mtp_draft_input_hiddens",
    ):
        value = getattr(model_input, name, None)
        if value is not None:
            setattr(model_input, name, value.index_select(0, rows))
    if model_input.multimodal_params is not None:
        model_input.multimodal_params = [
            params for params, keep in zip(model_input.multimodal_params, cpu_mask.tolist()) if keep
        ]
    model_input.batch_size = plan.dynamic_batch_size
    model_input.max_q_seq_len = 1
    return model_input, selected_cpu
