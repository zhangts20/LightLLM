import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo


@triton.jit
def _gen_rope_state(
    position_ids,
    cos_table,
    sin_table,
    out_positions,
    out_cos,
    out_sin,
    dim: tl.constexpr,
    size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    position = tl.load(position_ids + row)
    columns = tl.arange(0, BLOCK_D)
    cos = tl.load(cos_table + position.to(tl.int64) * dim + columns, columns < dim, other=0.0)
    sin = tl.load(sin_table + position.to(tl.int64) * dim + columns, columns < dim, other=0.0)
    for axis in tl.static_range(3):
        offset = (axis * size + row).to(tl.int64) * dim + columns
        tl.store(out_cos + offset, cos, columns < dim)
        tl.store(out_sin + offset, sin, columns < dim)
        tl.store(out_positions + axis * size + row, position)


def gen_3axis_rope_state(position_ids, cos_table, sin_table):
    """Expand continuous positions and gather both rotary tables in one launch."""
    size, dim = position_ids.numel(), cos_table.shape[1]
    positions = torch.empty((3, size), dtype=position_ids.dtype, device=position_ids.device)
    cos = torch.empty((3, size, dim), dtype=cos_table.dtype, device=cos_table.device)
    sin = torch.empty((3, size, dim), dtype=sin_table.dtype, device=sin_table.device)
    if size:
        _gen_rope_state[(size,)](
            position_ids,
            cos_table,
            sin_table,
            positions,
            cos,
            sin,
            dim,
            size,
            triton.next_power_of_2(dim),
            num_warps=4,
        )
    return positions, cos, sin


class MiniCPMV4_6InferStateInfo(Qwen2VLInferStateInfo):
    """MiniCPM-V-4.6 的位置编码状态：标准 1D RoPE，不使用 mrope。

    HF 的 `MiniCPMV4_6Model.forward` 把 `position_ids` 直接透传给 language_model
    （默认 None → 整条序列连续 1D 位置），图像特征通过 `masked_scatter` 注入
    inputs_embeds，对图像 token 不做任何位置偏移。因此这里把 mrope 的三轴位置
    统一成同一组连续位置：

    - 三轴取值相同，且 `mrope_section` 的 t/h/w mask 互斥且恰好覆盖全部旋转维度，
      于是 mrope kernel 退化为标准 RoPE，与 HF 完全一致；
    - 不再读取 multimodal_params 里的 `grid_thwd`（我们的 tokenizer 也不设置它），
      避免 qwen2_vl 的 mrope 路径因缺少图像布局信息而报错。
    """

    def init_some_extra_state(self, model):
        self.target_device = model.target_device
        rope_scaling = model.config.get("rope_scaling", {})
        self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))
        InferStateInfo.init_some_extra_state(self, model)

        if self.platform_backend.name == "maca":
            self.position_ids, self.position_cos, self.position_sin = gen_3axis_rope_state(
                self.position_ids, model._cos_cached, model._sin_cached
            )
        else:
            self.position_ids = self.position_ids.unsqueeze(0).expand(3, -1).contiguous()
            self.position_cos = model._cos_cached[self.position_ids]
            self.position_sin = model._sin_cached[self.position_ids]
        return
