import torch
import dataclasses
import triton
from lightllm.utils.envs_utils import get_added_mtp_kv_layer_num, get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.torch_dtype_utils import get_torch_dtype

logger = init_logger(__name__)


@dataclasses.dataclass
class LinearAttCacheConfig:
    tp_world_size: int
    # full att 的参数
    full_att_all_num_kv_heads: int
    full_att_dtype: torch.dtype
    full_att_num_kv_heads: int  # 这个是 tp 后的head头数量
    full_att_head_dim: int

    # linear att 的参数
    global_linear_k_heads: int
    global_linear_v_heads: int
    num_linear_k_heads: int
    num_linear_v_heads: int
    head_linear_k_dim: int
    head_linear_v_dim: int
    conv_kernel_size: int
    linear_layer_num: int
    conv_state_dtype: torch.dtype
    ssm_state_dtype: torch.dtype
    full_attention_interval: int
    all_layer_num: int  # 包括 linear att 和 full att 的层加起来的层数
    draft_full_att_kv_layer_num: int = 0
    # SenseNova Loop passes the execution-layer layout. Empty keeps the interval formulas.
    layer_types: tuple = ()

    def __post_init__(self):
        if not self.layer_types:
            self._layer_indices = ()
            return
        self.layer_types = tuple(self.layer_types)
        indices = []
        full_count = linear_count = 0
        for kind in self.layer_types:
            if kind == "linear_attention":
                indices.append(linear_count)
                linear_count += 1
            else:
                indices.append(full_count)
                full_count += 1
        # MTP draft layers only use full-attention KV slots after the main model.
        indices.extend(range(full_count, full_count + self.draft_full_att_kv_layer_num))
        self._layer_indices = tuple(indices)
        self.linear_layer_num = linear_count
        self.all_layer_num = len(self.layer_types)

    def get_linear_state_layer_index(self, layer_index: int) -> int:
        if self._layer_indices:
            return self._layer_indices[layer_index]
        return layer_index - (layer_index // self.full_attention_interval)

    def get_conv_dim(self):
        # 第一项对应q的参数，第二项对应k的参数，第三项对应v的参数
        # 由于 k_dim = q_dim, k_heads = q_heads, 所以第一项和第二项的计算
        # 形式相同，但是实际内在含义是不同的。
        return (
            self.head_linear_k_dim * self.num_linear_k_heads
            + self.head_linear_k_dim * self.num_linear_k_heads
            + self.head_linear_v_dim * self.num_linear_v_heads
        )

    def get_main_model_full_att_layer_num(self):
        full_att_layer_num = self.all_layer_num - self.linear_layer_num
        if not self._layer_indices:
            assert full_att_layer_num == self.all_layer_num // self.full_attention_interval
        return full_att_layer_num

    def get_full_att_kv_layer_num_with_draft_model(self):
        return self.get_main_model_full_att_layer_num() + self.draft_full_att_kv_layer_num

    def _hardware_platform(self) -> str:
        try:
            return get_env_start_args().hardware_platform
        except Exception:
            return ""

    def conv_state_feat_last(self) -> bool:
        return self._hardware_platform() in ("ascend", "mlu")

    def get_full_att_kv_layer_index(self, layer_index: int) -> int:
        """Map a global target/draft layer index to the packed full-attention cache."""

        if self._layer_indices:
            return self._layer_indices[layer_index]
        layer_index = int(layer_index)
        if layer_index >= self.all_layer_num:
            kv_layer_index = layer_index - self.linear_layer_num
        else:
            kv_layer_index = layer_index // self.full_attention_interval
        assert 0 <= kv_layer_index < self.get_full_att_kv_layer_num_with_draft_model(), (
            f"layer {layer_index} maps outside the packed full-attention KV cache: "
            f"slot={kv_layer_index}, slots={self.get_full_att_kv_layer_num_with_draft_model()}"
        )
        return kv_layer_index

    def get_conv_state_shape(self):
        # Base committed sliding-window state, without speculative MTP tail.
        dim = self.get_conv_dim()
        window = self.conv_kernel_size - 1
        return (window, dim) if self.conv_state_feat_last() else (dim, window)

    def get_mtp_conv_state_shape(self, mtp_step: int):
        # Working state with room for S speculative tokens before acceptance.
        dim = self.get_conv_dim()
        window = (self.conv_kernel_size - 1) + mtp_step
        return (window, dim) if self.conv_state_feat_last() else (dim, window)

    def get_ssm_state_shape(self):
        return (self.num_linear_v_heads, self.head_linear_k_dim, self.head_linear_v_dim)

    def get_conv_state_bytes_per_layer(self):
        return self.get_conv_dim() * (self.conv_kernel_size - 1) * self.conv_state_dtype.itemsize

    def get_ssm_state_bytes_per_layer(self):
        return self.num_linear_v_heads * self.head_linear_k_dim * self.head_linear_v_dim * self.ssm_state_dtype.itemsize

    def get_cpu_cache_big_page_bytes(self):
        a = self.get_cpu_cache_full_att_bytes()
        b = self.get_cpu_cache_conv_bytes()
        c = self.get_cpu_cache_ssm_bytes()

        return triton.cdiv(a + b + c, 16) * 16

    def get_cpu_cache_full_att_bytes(self):
        args = get_env_start_args()
        big_page_token_num = args.linear_att_page_block_num * args.linear_att_hash_page_size
        assert big_page_token_num == args.cpu_cache_token_page_size
        # 沐曦的 int8kv 需要额外存储 scale 数据，CUDA 和 NPU 暂不支持。
        if args.hardware_platform == "maca" and args.llm_kv_type == "int8kv":
            group = args.llm_kv_quant_group_size
            assert self.full_att_head_dim % group == 0
            per_token = (
                2 * self.full_att_all_num_kv_heads * self.full_att_head_dim
                + 2
                * self.full_att_all_num_kv_heads
                * (self.full_att_head_dim // group)
                * self.full_att_dtype.itemsize
            )
        else:
            per_token = 2 * self.full_att_all_num_kv_heads * self.full_att_head_dim * self.full_att_dtype.itemsize
        a = per_token * self.get_full_att_kv_layer_num_with_draft_model() * big_page_token_num
        return a

    def get_cpu_cache_conv_bytes(self):
        b = self.get_conv_state_bytes_per_layer() * self.linear_layer_num * self.tp_world_size
        return b

    def get_cpu_cache_ssm_bytes(self):
        c = self.get_ssm_state_bytes_per_layer() * self.linear_layer_num * self.tp_world_size
        return c

    @staticmethod
    def load_from_args() -> "LinearAttCacheConfig":
        args = get_env_start_args()
        model_path = args.model_dir
        from transformers.configuration_utils import PretrainedConfig

        model_cfg, _ = PretrainedConfig.get_config_dict(model_path)
        model_type = model_cfg["model_type"]
        assert model_type in [
            "qwen3_5",
            "sensenova_loop",
            "qwen3_5_moe",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        ]
        llm_config = model_cfg.get("text_config", model_cfg)
        if model_type == "sensenova_loop":
            from lightllm.models.sensenova_loop.config import get_loop_layer_types

            layer_types = get_loop_layer_types(model_cfg["loop_settings"], llm_config)
            n_layer = len(layer_types)
            interval = llm_config.get("full_attention_interval", 0)
        else:
            layer_types = ()
            n_layer = llm_config["num_hidden_layers"]
            interval = llm_config["full_attention_interval"]

        tp_world_size = args.tp // args.dp
        return LinearAttCacheConfig(
            tp_world_size=tp_world_size,
            full_att_all_num_kv_heads=llm_config["num_key_value_heads"],
            full_att_dtype=get_torch_dtype(args.data_type),
            full_att_num_kv_heads=max(1, llm_config["num_key_value_heads"] // tp_world_size),
            full_att_head_dim=llm_config["head_dim"],
            global_linear_k_heads=llm_config["linear_num_key_heads"],
            global_linear_v_heads=llm_config["linear_num_value_heads"],
            num_linear_k_heads=max(1, llm_config["linear_num_key_heads"] // tp_world_size),
            num_linear_v_heads=max(1, llm_config["linear_num_value_heads"] // tp_world_size),
            head_linear_k_dim=llm_config["linear_key_head_dim"],
            head_linear_v_dim=llm_config["linear_value_head_dim"],
            conv_kernel_size=llm_config["linear_conv_kernel_dim"],
            linear_layer_num=n_layer - (n_layer // interval) if interval else 0,
            conv_state_dtype=get_torch_dtype(args.data_type),
            ssm_state_dtype=get_torch_dtype(args.linear_att_ssm_data_type),
            full_attention_interval=interval,
            all_layer_num=n_layer,
            draft_full_att_kv_layer_num=get_added_mtp_kv_layer_num(),
            layer_types=tuple(layer_types),
        )
