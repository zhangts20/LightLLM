import torch

ACL_FORMAT_FRACTAL_NZ = 29


def format_dense_weight_nz(weight: torch.Tensor) -> torch.Tensor:
    import torch_npu

    if weight.shape in ((5120, 4120), (5120, 8704)):
        return torch_npu.npu_format_cast(weight.t().contiguous(), ACL_FORMAT_FRACTAL_NZ).t()
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
