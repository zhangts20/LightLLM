"""910B4 gating with rectangular loads, including partial token/head tiles.

Do not flatten (token, head) indexing with division/remainder: the installed
Triton-Ascend adapter lowers those masked gathers to unconditional scalar loads
followed by select, so a partial final tile can read beyond packed QKV storage.
Keep row/head axes explicit to obtain bounded subview copies in adapter MLIR.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _gating_910b4(
    G,
    BO,
    AL,
    A,
    B,
    BIAS,
    N: tl.constexpr,
    H: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    head = tl.arange(0, HEADS)
    al = tl.load(AL + head, head < H, other=0).to(tl.float32)
    bias = tl.load(BIAS + head, head < H, other=0).to(tl.float32)
    mask = (row[:, None] < N) & (head[None, :] < H)
    a = tl.load(A + row[:, None] * SA + head[None, :], mask, other=0).to(tl.float32)
    b = tl.load(B + row[:, None] * SB + head[None, :], mask, other=0).to(tl.float32)
    z = a + bias[None, :]
    softplus = tl.where(BETA * z <= THRESHOLD, (1 / BETA) * tl.log(1 + tl.exp(BETA * z)), z)
    g = -tl.exp(al)[None, :] * softplus
    # Preserve the historical BF16 rounding before writing the FP32 beta output.
    beta = tl.sigmoid(b).to(B.dtype.element_ty)
    tl.store(G + row[:, None] * H + head[None, :], g, mask)
    tl.store(BO + row[:, None] * H + head[None, :], beta, mask)


def fused_gdn_gating_910b4(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    rows: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, h = a.shape
    g = torch.empty((n, h), device=a.device, dtype=torch.float32)
    bo = torch.empty_like(g)
    _gating_910b4[(triton.cdiv(n, rows),)](
        g, bo, A_log, a, b, dt_bias, n, h, a.stride(0), b.stride(0), beta, threshold, rows, triton.next_power_of_2(h)
    )
    return g, bo
