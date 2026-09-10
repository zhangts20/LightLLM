from typing import Optional, Tuple

import torch
from .fused_gdn_gating_910b2c import fused_gdn_gating_910b2c


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    run_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (
        a.shape[0] >= 128
        and a.shape[1] == 12
        and a.dtype == torch.bfloat16
        and b.dtype == torch.bfloat16
        and a.stride(1) == b.stride(1) == 1
        and run_config is None
        and a.device.type == "npu"
        and torch.npu.get_device_name(a.device) == "Ascend910B4"
    ):
        from .fused_gdn_gating_910b4 import fused_gdn_gating_910b4

        return fused_gdn_gating_910b4(A_log, a, b, dt_bias, beta, threshold)

    return fused_gdn_gating_910b2c(A_log, a, b, dt_bias, beta, threshold, run_config=run_config)
