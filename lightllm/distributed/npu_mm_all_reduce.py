import torch
from typing import Any
from lightllm.utils.dist_utils import get_current_rank_in_dp
from lightllm.utils.envs_utils import get_env_start_args

# The minimum number of tokens to use npu_mm_all_reduce.
NPU_MM_ALL_REDUCE_MIN_TOKENS = 288
NPU_W8A8_MM_ALL_REDUCE_MIN_TOKENS = 288

# ProcessGroup -> hcom name
_HCOM_CACHE: dict = {}


def get_hccl_comm_name(group: Any) -> str:
    from lightllm.distributed.communication_op import CustomProcessGroup

    pg = group.device_group if isinstance(group, CustomProcessGroup) else group
    key = id(pg)
    cached = _HCOM_CACHE.get(key)
    if cached is not None:
        return cached

    backend = pg._get_backend(torch.device("npu"))
    if not hasattr(backend, "get_hccl_comm_name"):
        raise RuntimeError("ProcessGroupHCCL.get_hccl_comm_name is unavailable")
    try:
        import torch.distributed as dist

        rank = dist.get_rank(group=pg)
    except Exception:
        rank = get_current_rank_in_dp()
    name = backend.get_hccl_comm_name(rank)
    _HCOM_CACHE[key] = name

    return name


def get_mm_ar_weight(mm_weight) -> torch.Tensor:
    # COLMMWeight stores W as [out, in]; fused op needs x2 as [in, out] (= W.T).
    cached = getattr(mm_weight, "_npu_mm_ar_weight", None)
    w = mm_weight.mm_param.weight
    if cached is not None and cached.shape[0] == w.shape[1] and cached.shape[1] == w.shape[0]:
        return cached
    # Contiguous copy once, inference weights are static.
    mm_weight._npu_mm_ar_weight = w.t().contiguous()
    return mm_weight._npu_mm_ar_weight


def can_use_npu_mm_all_reduce(
    token_num: int,
    infer_state,
    min_tokens: int = NPU_MM_ALL_REDUCE_MIN_TOKENS,
) -> bool:
    # Check whether to fuse matmul and all_reduce into a single operation.
    args = get_env_start_args()
    if getattr(args, "disable_npu_mm_all_reduce", False):
        return False

    if getattr(args, "hardware_platform", "cuda") != "ascend":
        return False

    if getattr(args, "enable_tpsp_mix_mode", False):
        return False

    if not getattr(infer_state, "is_prefill", False):
        return False

    if token_num < min_tokens:
        return False

    tp_world_size = getattr(args, "tp", 1) // max(getattr(args, "dp", 1), 1)
    if tp_world_size <= 1:
        return False

    return True


def can_use_npu_w8a8_mm_all_reduce(token_num: int, infer_state) -> bool:
    return can_use_npu_mm_all_reduce(
        token_num,
        infer_state,
        min_tokens=NPU_W8A8_MM_ALL_REDUCE_MIN_TOKENS,
    )


def is_plain_mm_weight(mm_weight) -> bool:
    # Check whether the weight is a no quantization mm weight.
    try:
        method = mm_weight.quant_method
        name = getattr(method, "method_name", None)
        if name is not None and name != "none":
            return False
        w = mm_weight.mm_param.weight
        if w.dtype not in (torch.float16, torch.bfloat16):
            return False
        if getattr(mm_weight.mm_param, "weight_scale", None) is not None:
            return False
        return True
    except Exception:
        return False


def is_npu_w8a8_weight(mm_weight) -> bool:
    try:
        return (
            mm_weight.quant_method.method_name == "w8a8-ascend"
            and mm_weight.mm_param.weight.dtype == torch.int8
            and mm_weight.mm_param.weight_scale is not None
            and mm_weight.bias is None
        )
    except Exception:
        return False


def npu_mm_all_reduce(
    x: torch.Tensor,
    mm_weight,
    infer_state,
    *,
    reduce_op: str = "sum",
) -> torch.Tensor:
    # Perform all_reduce(x @ W.T) via torch_npu.npu_mm_all_reduce_base.
    import torch_npu

    hcom = get_hccl_comm_name(infer_state.dist_group)
    w = get_mm_ar_weight(mm_weight)
    out = torch_npu.npu_mm_all_reduce_base(x, w, hcom, reduce_op=reduce_op)
    bias = getattr(mm_weight, "bias", None)
    if bias is not None:
        out = out + bias

    return out


def npu_w8a8_mm_all_reduce(
    x: torch.Tensor,
    mm_weight,
    infer_state,
    *,
    reduce_op: str = "sum",
) -> torch.Tensor:
    import torch_npu

    hcom = get_hccl_comm_name(infer_state.dist_group)
    x_q, x_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)

    dequant_scale = getattr(mm_weight, "_npu_mm_ar_dequant_scale", None)
    expected_dtype = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32
    if dequant_scale is None or dequant_scale.dtype != expected_dtype:
        dequant_scale = mm_weight.mm_param.weight_scale.to(expected_dtype)
        mm_weight._npu_mm_ar_dequant_scale = dequant_scale

    return torch_npu.npu_mm_all_reduce_base(
        x_q,
        mm_weight.mm_param.weight,
        hcom,
        reduce_op=reduce_op,
        dequant_scale=dequant_scale,
        pertoken_scale=x_scale,
    )
