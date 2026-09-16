from lightllm.common.quantization.quantize_method import QuantizationMethod
from .triton_impl import FuseMoeTriton
from .marlin_impl import FuseMoeMarlin
from .deepgemm_impl import FuseMoeDeepGEMM
from .npu_impl import FuseMoeNPU
from .npu_w8a8_impl import FuseMoeNPUW8A8


def select_fuse_moe_impl(quant_method: QuantizationMethod, enable_ep_moe: bool):
    if quant_method.method_name == "w8a8-ascend":
        if enable_ep_moe:
            raise NotImplementedError("Ascend W8A8 fused MoE does not support expert parallelism yet")
        return FuseMoeNPUW8A8

    if quant_method.method_name == "none" and quant_method.target_device.type == "npu":
        if enable_ep_moe:
            raise NotImplementedError("Ascend native floating-point fused MoE does not support expert parallelism yet")
        return FuseMoeNPU

    if enable_ep_moe:
        return FuseMoeDeepGEMM

    if quant_method.method_name == "awq_marlin":
        return FuseMoeMarlin
    else:
        return FuseMoeTriton
