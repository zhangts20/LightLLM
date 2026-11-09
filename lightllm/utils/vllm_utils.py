import torch
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        torch.cuda.init()
    from vllm import _custom_ops as ops

    vllm_ops = ops
    HAS_VLLM = True

except:
    HAS_VLLM = False
    cutlass_scaled_mm = None
    vllm_ops = None
    logger.warning(
        "vllm is not installed, you can't use the api of it. \
                   You can solve it by running `pip install vllm`."
    )


def _cutlass_scaled_mm_out(out, a, b, scale_a, scale_b, bias=None):
    try:
        return torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
    except AttributeError:
        pass

    from vllm_metax.model_executor.layers.quantization._python_api_ops import cutlass_scaled_mm as maca_cutlass_scaled_mm

    if scale_b.dim() == 1:
        scale_b = scale_b.reshape(-1, 1)

    result = maca_cutlass_scaled_mm(a, b, scale_a, scale_b, out.dtype, bias)
    out.copy_(result)
    return out


cutlass_scaled_mm = _cutlass_scaled_mm_out
