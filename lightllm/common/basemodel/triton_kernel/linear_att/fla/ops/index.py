# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    lens = prepare_lens(cu_seqlens)
    n_chunks = torch.div(lens + (chunk_size - 1), chunk_size, rounding_mode="floor")
    device = cu_seqlens.device
    dtype = cu_seqlens.dtype
    batch = n_chunks.numel()
    if batch == 0:
        return cu_seqlens.new_empty((0, 2))

    offsets = torch.empty(batch + 1, device=device, dtype=dtype)
    offsets[0] = 0
    torch.cumsum(n_chunks, dim=0, out=offsets[1:])

    total = int(offsets[-1].item())
    if total == 0:
        return cu_seqlens.new_empty((0, 2))

    indices = torch.arange(total, device=device, dtype=dtype)
    batch_ids = torch.searchsorted(offsets[1:], indices, right=True)
    local_ids = indices - offsets[batch_ids]
    return torch.stack((batch_ids, local_ids), dim=1)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)
