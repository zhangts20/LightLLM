"""MiniCPM-V-4.6 视觉塔（lightllm visual server 侧）。

结构（与 HF `MiniCPMV4_6VisionModel` + `MiniCPMV4_6Merger` 一一对应）：

    vision_tower.embeddings        Conv2d patch_embedding(3->1152, 14, stride 14)
                                   + position_embedding(70*70, 1152)  最近邻查表
    vision_tower.encoder.layers    27 层 pre-LN encoder，双向全注意力（段内）
      └ layer_index == 6 之后插入 vit_merger：
            window 2x2 自注意力(LN 后) + 残差(窗口 4 patch 均值)
            → pre_norm(4608) → linear_1(4608->17216) → gelu_tanh → linear_2(17216->1152)
            → 序列长度 4 倍下采样
    vision_tower.post_layernorm
    merger.mlp.0                   pre_norm(4608) → linear_1(4608->4608) → GELU
                                   → linear_2(4608->1024)，再下采样 4 倍

段(segment)语义：`target_sizes` 的每一行是一个独立的 NaViT 段（一张图的全局视图，
以及该图的每个切片各为一段），段内双向注意力、段间互不可见。HF 的
`MiniCPMV4_6Merger` 就是按行循环处理，故两者在无切片/单行场景下完全等价，
且在多图、含切片的批处理下同样正确（HF 的 vit_merger 依赖“批内各图 patch 数
相同”的 reshape 假设，这里改成逐段处理，避免该隐含约束）。
"""

import os
import json
from io import BytesIO
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers.activations import ACT2FN

from lightllm.models.visual_utils import VisualDeviceMixin
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.server.multimodal_params import ImageItem
from lightllm.server.visualserver import get_vit_attn_backend
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

VISION_TOKEN_DIVISOR = 16


class MiniCPMV4_6VisionEmbeddings(nn.Module):
    def __init__(self, hidden_size=1152, patch_size=14, num_channels=3, image_size=980):
        super().__init__()
        self.embed_dim = hidden_size
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )
        self.num_patches_per_side = image_size // patch_size
        self.position_embedding = nn.Embedding(self.num_patches_per_side ** 2, hidden_size)

    def nearest_position_ids(self, target_sizes: torch.Tensor) -> torch.Tensor:
        """与 HF `MiniCPMV4_6VisionEmbeddings.forward` 的位置查表逐字一致。

        两个都必须照抄，否则行/列的坐标会落在桶边界的另一侧，取到相差 1 的位置编码：
          1) 坐标用 `arange(0, 1 - 1e-6, 1/n)` 生成，不能写成 `arange(n) / n`；
          2) 步长必须写成 `1 / 0-dim int32 tensor`（得到 float32 张量步长）。
             torch.arange 在“张量步长”与“python float 步长”两条路径上的累加精度不同：
             实测 n=100 时第 10 列坐标，前者 0.0999999940（桶 6），后者 0.1000000015
             （恰等于 7/70 边界，right=True 取桶 7）。
        """
        side = self.num_patches_per_side
        boundaries = torch.arange(1 / side, 1.0, 1 / side)
        pos_ids = []
        for height, width in target_sizes.tolist():
            frac_h = torch.arange(0, 1 - 1e-6, 1 / torch.tensor(height, dtype=torch.int32))
            frac_w = torch.arange(0, 1 - 1e-6, 1 / torch.tensor(width, dtype=torch.int32))
            bucket_h = torch.bucketize(frac_h, boundaries, right=True)
            bucket_w = torch.bucketize(frac_w, boundaries, right=True)
            pos_ids.append((bucket_h[:, None] * side + bucket_w).flatten())
        return torch.cat(pos_ids).to(self.position_embedding.weight.device)

    def forward(self, pixel_values: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        # pixel_values: [1, 3, patch_size, sum(h*w)*patch_size]（NaViT 打包）
        # 返回 [sum(h*w), D]，后续所有模块内部都不带 batch 维
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        pos_ids = self.nearest_position_ids(target_sizes)
        embeddings = embeddings + self.position_embedding(pos_ids).unsqueeze(0)
        return embeddings[0]


class MiniCPMV4_6VisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class MiniCPMV4_6VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        attn_output = torch.empty_like(q)
        get_vit_attn_backend()(q.contiguous(), k.contiguous(), v.contiguous(), attn_output, cu_seqlens, max_seqlen)
        attn_output = attn_output.reshape(seq_length, self.dim)
        return self.out_proj(attn_output)


class MiniCPMV4_6VisionEncoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, hidden_act, layer_norm_eps):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = MiniCPMV4_6VisionAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MiniCPMV4_6VisionMLP(hidden_size, intermediate_size, hidden_act)

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniCPMV4_6VisionEncoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, hidden_act, layer_norm_eps, num_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MiniCPMV4_6VisionEncoderLayer(hidden_size, intermediate_size, num_heads, hidden_act, layer_norm_eps)
                for _ in range(num_hidden_layers)
            ]
        )


class MiniCPMV4_6ViTWindowAttentionMerger(nn.Module):
    """第 6 层后的序列 4 倍下采样（2x2 window 注意力 + 2x2 拼接 MLP）。"""

    def __init__(self, hidden_size, intermediate_size, num_heads, layer_norm_eps, window_kernel_size=(2, 2)):
        super().__init__()
        self.window_kernel_size = tuple(window_kernel_size)
        self.embed_dim = hidden_size
        self.window_hidden_size = hidden_size * self.window_kernel_size[0] * self.window_kernel_size[1]
        self.window_intermediate_size = intermediate_size * self.window_kernel_size[0] * self.window_kernel_size[1]

        self.self_attn = MiniCPMV4_6VisionAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.pre_norm = nn.LayerNorm(self.window_hidden_size, eps=layer_norm_eps)
        self.linear_1 = nn.Linear(self.window_hidden_size, self.window_intermediate_size, bias=True)
        self.act = ACT2FN["gelu_pytorch_tanh"]
        self.linear_2 = nn.Linear(self.window_intermediate_size, hidden_size, bias=True)

    def build_window_index(self, target_sizes: torch.Tensor) -> torch.Tensor:
        """段内按不重叠 window 重排的 gather 索引，等价于 get_vision_window_index。

        全局视图/切片的 patch 网格边长均为 patch_size*4/14=4 的整数倍（偶数），
        因此窗口恰好铺满且无 padding，窗口数 = sum(h*w)/4。
        """
        device = target_sizes.device
        window_h, window_w = self.window_kernel_size
        indices = []
        offset = 0
        for height, width in target_sizes.tolist():
            merged_h, merged_w = height // window_h, width // window_w
            idx = torch.arange(height * width, device=device, dtype=torch.long)
            idx = idx.view(merged_h, window_h, merged_w, window_w).permute(0, 2, 1, 3).reshape(-1)
            indices.append(idx + offset)
            offset += height * width
        return torch.cat(indices) if indices else torch.empty(0, device=device, dtype=torch.long)

    def forward(self, hidden_states: torch.Tensor, target_sizes: torch.Tensor):
        # hidden_states: [T, D]
        window_h, window_w = self.window_kernel_size
        window_index = self.build_window_index(target_sizes)
        num_tokens = hidden_states.shape[0]

        gathered = hidden_states.index_select(0, window_index)
        gathered = self.layer_norm1(gathered)
        cu_seqlens = torch.arange(
            0,
            num_tokens + 1,
            window_h * window_w,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        attn_out = self.self_attn(gathered, cu_seqlens=cu_seqlens, max_seqlen=window_h * window_w)
        attn_full = torch.empty_like(attn_out).index_copy(0, window_index, attn_out)
        hidden_states = hidden_states + attn_full

        # 逐段做拼接 MLP（与 HF 完全一致）。不要合成一个大批次：bf16 GEMM 的分批方式
        # （1400 行一次 vs 5×280 行）会让累加顺序不同，实测相对误差 ~0.7%，虽数值等价但
        # 无法与 HF 逐位对齐，逐段调用代价也可忽略。
        outputs = []
        offset = 0
        for height, width in target_sizes.tolist():
            merged_h, merged_w = height // window_h, width // window_w
            seg = hidden_states[offset : offset + height * width]
            patch_5d = seg.view(merged_h, window_h, merged_w, window_w, self.embed_dim).permute(0, 2, 1, 3, 4)
            flat = patch_5d.reshape(merged_h * merged_w, self.window_hidden_size)
            residual = patch_5d.reshape(merged_h * merged_w, window_h * window_w, self.embed_dim).mean(dim=1)
            hidden_state = self.linear_2(self.act(self.linear_1(self.pre_norm(flat))))
            outputs.append(hidden_state + residual)
            offset += height * width
        return torch.cat(outputs)


class MiniCPMV4_6VisionTower(nn.Module):
    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=27,
        num_channels=3,
        patch_size=14,
        image_size=980,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        insert_layer_id=6,
        window_kernel_size=(2, 2),
        use_vit_merger=True,
    ):
        super().__init__()
        self.insert_layer_id = insert_layer_id
        self.use_vit_merger = use_vit_merger
        self.embeddings = MiniCPMV4_6VisionEmbeddings(hidden_size, patch_size, num_channels, image_size)
        self.encoder = MiniCPMV4_6VisionEncoder(
            hidden_size, intermediate_size, num_attention_heads, hidden_act, layer_norm_eps, num_hidden_layers
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.vit_merger = MiniCPMV4_6ViTWindowAttentionMerger(
            hidden_size, intermediate_size, num_attention_heads, layer_norm_eps, window_kernel_size
        )

    @staticmethod
    def _segment_cu_seqlens(target_sizes: torch.Tensor, device) -> torch.Tensor:
        lengths = target_sizes[:, 0] * target_sizes[:, 1]
        cu_seqlens = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32).to(device), (1, 0), value=0)
        return cu_seqlens

    def forward(self, pixel_values: torch.Tensor, target_sizes: torch.Tensor):
        """返回 (last_hidden_state[N, D], 下采样后的 target_sizes)。

        第二个返回值即 `MiniCPMV4_6Merger` 期望的 target_sizes（16x 模式下为
        原 target_sizes // 2），调用方不需要再除。
        """
        hidden_states = self.embeddings(pixel_values, target_sizes)
        device = hidden_states.device

        cu_seqlens = self._segment_cu_seqlens(target_sizes, device)
        max_seqlens = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        insert_layer_id = self.insert_layer_id if self.use_vit_merger else -1
        for layer_index, layer in enumerate(self.encoder.layers):
            hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlens)
            if layer_index == insert_layer_id:
                hidden_states = self.vit_merger(hidden_states, target_sizes)
                # 与 HF get_downsampled_inputs 一致：段长 4 倍下采样后重算 cu_seqlens
                target_sizes = target_sizes // 2
                cu_seqlens = self._segment_cu_seqlens(target_sizes, device)
                max_seqlens = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states, target_sizes


class MiniCPMV4_6DownsampleMLP(nn.Module):
    def __init__(self, hidden_size, llm_embed_dim):
        super().__init__()
        merged_hidden_size = hidden_size * 4
        self.pre_norm = nn.LayerNorm(merged_hidden_size, eps=1e-6)
        self.linear_1 = nn.Linear(merged_hidden_size, merged_hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(merged_hidden_size, llm_embed_dim, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.pre_norm(hidden_states).view(-1, self.linear_1.in_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)


class MiniCPMV4_6MergerMLP(nn.Module):
    """`merger_times=1`：先 2x2 拼接再做 MLP 投影到 LLM 维度。"""

    def __init__(self, hidden_size, llm_embed_dim, merge_kernel_size=(2, 2)):
        super().__init__()
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.mlp = nn.ModuleList([MiniCPMV4_6DownsampleMLP(hidden_size, llm_embed_dim)])

    def forward(self, hidden_states: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        merge_h, merge_w = self.merge_kernel_size
        embed_dim = hidden_states.shape[-1]
        outputs = []
        offset = 0
        for height, width in target_sizes.tolist():
            num_patches = height * width
            merged_h, merged_w = height // merge_h, width // merge_w
            seg = (
                hidden_states[offset : offset + num_patches]
                .view(merged_h, merge_h, merged_w, merge_w, embed_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(merged_h * merged_w, merge_h * merge_w * embed_dim)
            )
            outputs.append(self.mlp[0](seg))
            offset += num_patches
        return torch.cat(outputs)


class MiniCPMV4_6VisionModel(VisualDeviceMixin, nn.Module):
    """visual server 侧入口：负责预处理 + 前向 + 输出视觉 embeds。"""

    def __init__(self, kvargs, vision_config: dict, text_hidden_size: int, downsample_mode: str = "16x"):
        super().__init__()
        self.data_type = kvargs.get("data_type", "bfloat16")
        self.downsample_mode = downsample_mode
        self.use_vit_merger = downsample_mode != "4x"
        self.token_divisor = VISION_TOKEN_DIVISOR if self.use_vit_merger else 4
        hidden_size = vision_config.get("hidden_size", 1152)
        self.vision_tower = MiniCPMV4_6VisionTower(
            hidden_size=hidden_size,
            intermediate_size=vision_config.get("intermediate_size", 4304),
            num_attention_heads=vision_config.get("num_attention_heads", 16),
            num_hidden_layers=vision_config.get("num_hidden_layers", 27),
            num_channels=vision_config.get("num_channels", 3),
            patch_size=vision_config.get("patch_size", 14),
            image_size=vision_config.get("image_size", 980),
            hidden_act=vision_config.get("hidden_act", "gelu_pytorch_tanh"),
            layer_norm_eps=vision_config.get("layer_norm_eps", 1e-6),
            insert_layer_id=vision_config.get("insert_layer_id", 6),
            window_kernel_size=vision_config.get("window_kernel_size", (2, 2)),
            use_vit_merger=self.use_vit_merger,
        )
        self.merger = MiniCPMV4_6MergerMLP(hidden_size, text_hidden_size)
        self._init_datatype()

    def _init_datatype(self):
        if isinstance(self.data_type, torch.dtype):
            return
        if self.data_type in ["fp16", "float16"]:
            self.data_type = torch.float16
        elif self.data_type in ["bf16", "bfloat16"]:
            self.data_type = torch.bfloat16
        elif self.data_type in ["fp32", "float32"]:
            self.data_type = torch.float32
        else:
            raise ValueError(f"Unsupported datatype {self.data_type}!")
        return

    def load_model(self, weight_dir):
        from transformers import AutoProcessor

        # checkpoint 为 bf16，先把参数张量转成目标 dtype，load_state_dict 才不会被转回 fp32
        self.to(self.data_type)
        self.processor = AutoProcessor.from_pretrained(weight_dir).image_processor

        weight_dict = {}
        for file_ in sorted(os.listdir(weight_dir)):
            if not file_.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(weight_dir, file_), "pt", "cpu") as f:
                for k in f.keys():
                    if k.startswith("model.vision_tower.") or k.startswith("model.merger."):
                        weight_dict[k[len("model.") :]] = f.get_tensor(k)
        if not weight_dict:
            raise RuntimeError(f"no vision weights found in {weight_dir}")
        missing, unexpected = self.load_state_dict(weight_dict, strict=False)
        if missing:
            raise RuntimeError(f"missing vision weights: {missing[:8]} (total {len(missing)})")
        if unexpected:
            raise RuntimeError(f"unexpected vision weights: {unexpected[:8]} (total {len(unexpected)})")
        logger.info(f"MiniCPM-V-4.6 vision tower loaded, {len(weight_dict)} tensors from {weight_dir}")
        return

    @torch.no_grad()
    def encode(self, images: List[ImageItem]):
        pixel_values_list = []
        target_sizes_list = []
        uuids = []
        valid_ids = []
        valid_id = 0
        for img in images:
            if not isinstance(img, ImageItem):
                raise Exception("Unsupported input types: {} for {}".format(type(img), img))
            uuids.append(img.uuid)
            image_data = read_shm(get_shm_name_data(img.uuid))
            image = Image.open(BytesIO(image_data)).convert("RGB")
            out = self.processor(image, return_tensors="pt")
            pixel_values_list.append(out["pixel_values"])
            target_sizes = out["target_sizes"]
            target_sizes_list.append(target_sizes)

            token_num = int(target_sizes.prod(-1).sum().item()) // self.token_divisor
            if img.token_num is not None and token_num != img.token_num:
                raise ValueError(
                    f"image token num mismatch: processor={token_num} vs tokenizer={img.token_num} "
                    f"(image {img.image_w}x{img.image_h})"
                )
            valid_ids.append([valid_id, valid_id + token_num])
            valid_id += token_num

        if len(pixel_values_list) == 0:
            return None

        pixel_values, target_sizes = self.move_to_infer_device(
            torch.cat(pixel_values_list, dim=-1),
            torch.cat(target_sizes_list, dim=0),
            dtype=(self.data_type, None),
        )

        hidden_states, target_sizes = self.vision_tower(pixel_values, target_sizes)
        all_img_embeds = self.merger(hidden_states, target_sizes)

        return all_img_embeds, uuids, valid_ids
