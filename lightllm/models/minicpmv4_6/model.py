from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.models.qwen3_5.layer_weights.pre_and_post_layer_weight import (
    Qwen35PreAndPostLayerWeight,
)
from lightllm.models.qwen3_5.layer_infer.transformer_layer_infer import (
    Qwen35TransformerLayerInfer,
)
from lightllm.models.minicpmv4_6.infer_struct import MiniCPMV4_6InferStateInfo


@ModelRegistry(["minicpmv4_6"], is_multimodal=True)
class MiniCPMV4_6TpPartModel(Qwen3_5TpPartModel):
    """MiniCPM-V-4.6

    文本主干 config.json 的 text_config.model_type == "qwen3_5_text"，与 Qwen3.5 的
    混合注意力主干 (Gated DeltaNet + 每 full_attention_interval 层一个 full attention)
    完全一致，因此直接复用 qwen3_5 的权重类与推理类。

    checkpoint 中语言侧权重前缀为 model.language_model.*，由
    Qwen35PreAndPostLayerWeight 的 rename_weight_keys 原地改写为 model.*，
    与 Qwen35TransformerLayerWeight 期望的名字一致。

    视觉侧权重 (model.vision_tower.* / model.merger.*) 由 visual server 的
    MiniCPMV4_6VisionModel 加载（models/minicpmv4_6/visual.py）。

    位置编码沿用 MiniCPMV4_6InferStateInfo：HF 对整条序列（含图像占位 token）
    使用连续 1D 位置，故这里不走 qwen2_vl 的 mrope 路径。
    """

    transformer_weight_class = Qwen35TransformerLayerWeight
    pre_and_post_weight_class = Qwen35PreAndPostLayerWeight
    transformer_layer_infer_class = Qwen35TransformerLayerInfer
    infer_state_class = MiniCPMV4_6InferStateInfo
