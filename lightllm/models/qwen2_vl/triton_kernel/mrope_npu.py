import torch
from .mrope_npu_910b2c import can_use_mrope_prefill, mrope_small
from .mrope_npu_910b2c import mrope_prefill as _legacy_prefill
from .mrope_npu_910b4 import mrope_prefill as _910b4_prefill

@torch.no_grad()
def mrope_prefill(q, k, cos, sin, mrope_section, is_interleaved=True, partial_rotary_factor=0.25):
    impl = _910b4_prefill if torch.npu.get_device_name(q.device) == "Ascend910B4" else _legacy_prefill
    return impl(q, k, cos, sin, mrope_section, is_interleaved, partial_rotary_factor)
