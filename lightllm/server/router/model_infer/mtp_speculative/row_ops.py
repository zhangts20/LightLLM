"""Row operations with CUDA/MACA kernels and portable accelerator fallbacks."""

import torch


def select_accepted_tail_rows(b_req_mtp_start_loc, accept_len, **tensors):
    from lightllm.common.basemodel.triton_kernel.select_mtp_rows import (
        SelectedMtpRows,
        select_accepted_tail_rows as select_kernel,
    )

    if tensors["input_ids"].is_cuda:
        return select_kernel(
            b_req_mtp_start_loc=b_req_mtp_start_loc, accept_len=accept_len, **tensors
        )
    rows = (b_req_mtp_start_loc + accept_len - 1).long()
    return SelectedMtpRows(**{name: value.index_select(0, rows) for name, value in tensors.items()})


def build_chained_mtp_decode_input_inplace(input_ids, draft_token_ids, b_req_mtp_start_loc, accept_len):
    if input_ids.is_cuda:
        from lightllm.common.basemodel.triton_kernel.build_chained_mtp_decode_input import (
            build_chained_mtp_decode_input_inplace as build_kernel,
        )

        return build_kernel(input_ids, draft_token_ids, b_req_mtp_start_loc, accept_len)
    if not b_req_mtp_start_loc.numel():
        return draft_token_ids
    rows = torch.arange(input_ids.numel(), device=input_ids.device)
    request = torch.searchsorted(b_req_mtp_start_loc.long(), rows, right=True) - 1
    tails = b_req_mtp_start_loc.long() + accept_len.long() - 1
    shift = rows < tails.index_select(0, request)
    # Every shifted row precedes its request's accepted tail, so row + 1 exists.
    shifted_rows = rows[shift]
    draft_token_ids[shifted_rows] = input_ids.index_select(0, shifted_rows + 1)
    return draft_token_ids
