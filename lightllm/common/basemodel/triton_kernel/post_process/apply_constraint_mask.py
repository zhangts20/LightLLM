from lightllm.platform import get_backend
import torch
import triton
import triton.language as tl

MASKED_LOGIT = -1000000.0


@triton.jit
def _apply_constraint_mask(
    logits,
    req_to_bitmask,
    req_to_bitmask_enabled,
    logits_stride,
    mask_req_stride,
    VOCAB_SIZE: tl.constexpr,
    LOGITS_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MASKED_LOGIT: tl.constexpr,
):
    row = tl.program_id(0)
    token_ids = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    logits_ptr = logits + row * logits_stride + token_ids

    if tl.load(req_to_bitmask_enabled + row):
        mask_ptr = req_to_bitmask + row * mask_req_stride
        word_ids = tl.program_id(1) * (BLOCK_SIZE // 32) + tl.arange(0, BLOCK_SIZE // 32)
        mask_words = tl.load(mask_ptr + word_ids, mask=word_ids < tl.cdiv(VOCAB_SIZE, 32), other=0)
        allowed = ((mask_words[:, None] >> tl.arange(0, 32)[None, :]) & 1).reshape(BLOCK_SIZE)
        tl.store(
            logits_ptr,
            MASKED_LOGIT,
            mask=(token_ids < LOGITS_WIDTH) & ((allowed == 0) | (token_ids >= VOCAB_SIZE)),
        )
    elif LOGITS_WIDTH > VOCAB_SIZE:
        tl.store(logits_ptr, MASKED_LOGIT, mask=(token_ids >= VOCAB_SIZE) & (token_ids < LOGITS_WIDTH))


def apply_constraint_mask(
    logits: torch.Tensor,
    req_to_bitmask: torch.Tensor,
    req_to_bitmask_enabled: torch.Tensor,
    vocab_size: int,
) -> None:
    assert get_backend().runtime.is_device_tensor(logits) and logits.stride(1) == 1
    assert get_backend().runtime.is_device_tensor(req_to_bitmask) and get_backend().runtime.is_device_tensor(
        req_to_bitmask_enabled
    )
    if logits.shape[0] == 0:
        return
    block_size = 4096
    _apply_constraint_mask[(logits.shape[0], triton.cdiv(logits.shape[1], block_size))](
        logits,
        req_to_bitmask,
        req_to_bitmask_enabled,
        logits.stride(0),
        req_to_bitmask.stride(0),
        VOCAB_SIZE=vocab_size,
        LOGITS_WIDTH=logits.shape[1],
        BLOCK_SIZE=block_size,
        MASKED_LOGIT=MASKED_LOGIT,
    )
