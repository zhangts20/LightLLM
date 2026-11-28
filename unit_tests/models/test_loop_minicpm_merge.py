"""Synthetic configuration checks; no checkpoint, processor or device is loaded."""
import ast
import importlib.util
import sys
import typing
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

ROOT = Path(__file__).resolve().parents[2]


def load_source(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mock_package(monkeypatch, name):
    package = ModuleType(name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, name, package)


@pytest.fixture
def modules(monkeypatch):
    config = load_source("lightllm/utils/config_utils.py", "lightllm.utils.merge_config_under_test")
    cache = load_source("lightllm/common/linear_att_cache_manager/config_objs.py", "merge_cache_config_under_test")
    mock_package(monkeypatch, "lightllm.models")
    mock_package(monkeypatch, "lightllm.models.sensenova_loop")
    loop = load_source("lightllm/models/sensenova_loop/config.py", "loop_layout_under_test")
    monkeypatch.setitem(sys.modules, "lightllm.models.sensenova_loop.config", loop)
    return config, cache


def model_config(model_type):
    return {
        "model_type": model_type,
        "architectures": ["SyntheticModel"],
        "eos_token_id": 9,
        "text_config": {
            "num_hidden_layers": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "full_attention_interval": 4,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        },
        "loop_settings": {"start": 1, "end": 4, "eval_count": 2},
    }


@pytest.mark.parametrize("model_type", ["minicpmv4_6", "sensenova_loop"])
@pytest.mark.parametrize("platform", ["cuda", "maca", "mlu", "ascend"])
def test_cache_loading_preserves_both_execution_layouts(monkeypatch, modules, model_type, platform):
    _, cache = modules
    cfg = model_config(model_type)
    monkeypatch.setattr(PretrainedConfig, "get_config_dict", lambda *args, **kwargs: (cfg, {}))
    args = SimpleNamespace(
        model_dir="synthetic", tp=4, dp=2, data_type="bf16", linear_att_ssm_data_type="fp32", hardware_platform=platform
    )
    monkeypatch.setattr(cache, "get_env_start_args", lambda: args)
    monkeypatch.setattr(cache, "get_added_mtp_kv_layer_num", lambda: 1)

    result = cache.LinearAttCacheConfig.load_from_args()

    assert result.tp_world_size == 2
    assert result.full_att_num_kv_heads == 2
    assert result.conv_state_dtype == torch.bfloat16
    assert result.ssm_state_dtype == torch.float32
    if model_type == "sensenova_loop":
        assert (result.all_layer_num, result.linear_layer_num) == (7, 5)
        assert [result.get_full_att_kv_layer_index(i) for i in (3, 6, 7)] == [0, 1, 2]
        assert [result.get_linear_state_layer_index(i) for i in (0, 1, 2, 4, 5)] == [0, 1, 2, 3, 4]
    else:
        assert (result.all_layer_num, result.linear_layer_num) == (4, 3)
        assert [result.get_full_att_kv_layer_index(i) for i in (3, 4)] == [0, 1]
        assert result.layer_types == ()
    expected = (3, 64) if platform in ("mlu", "ascend") else (64, 3)
    assert result.get_conv_state_shape() == expected
    assert result.get_mtp_conv_state_shape(2) == ((5, 64) if platform in ("mlu", "ascend") else (64, 5))


@pytest.mark.parametrize("model_type", ["minicpmv4_6", "sensenova_loop", "llava"])
def test_model_metadata_retains_tokenizer_stop_ids(monkeypatch, modules, model_type, tmp_path):
    config, _ = modules
    cfg = model_config(model_type)
    monkeypatch.setattr(config, "get_config_json", lambda path: cfg)
    monkeypatch.setattr(PretrainedConfig, "get_config_dict", lambda *args, **kwargs: (cfg, {}))
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(eos_token_id=77)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert set(config.get_eos_token_ids(str(tmp_path))) == {9, 77}
    if model_type != "llava":
        assert config.is_linear_att_mixed_model(str(tmp_path)) is True
        assert config.get_tool_call_parser_for_model(str(tmp_path)) == "qwen3_coder"


@pytest.mark.parametrize("model_type", ["minicpmv4_6", "sensenova_loop"])
def test_tokenizer_uses_the_correct_processor_without_loading_models(monkeypatch, model_type):
    # Execute the real factory function with I/O dependencies mocked, excluding
    # module-level imports of unrelated inference kernels and model classes.
    tree = ast.parse((ROOT / "lightllm/server/tokenizer.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_tokenizer")
    calls = []

    class FastTokenizer:
        pass

    tokenizer = FastTokenizer()
    config = model_config(model_type)
    config["vision_config"] = {}
    fast_factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    namespace = {
        "__name__": "lightllm.server.tokenizer",
        "__package__": "lightllm.server",
        "Union": typing.Union,
        "PreTrainedTokenizer": object,
        "PreTrainedTokenizerFast": FastTokenizer,
        "AutoTokenizer": fast_factory,
        "PretrainedConfig": SimpleNamespace(get_config_dict=lambda *args: (config, {})),
        "logger": SimpleNamespace(info=lambda *args: None, warning=lambda *args: None),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "tokenizer_factory_under_test", "exec"), namespace)

    def processor(kind):
        def load(*args, **kwargs):
            calls.append(kind)
            return SimpleNamespace(image_processor="mini_image") if kind == "AutoProcessor" else "loop_image"

        return SimpleNamespace(from_pretrained=load)

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = processor("AutoProcessor")
    transformers.AutoImageProcessor = processor("AutoImageProcessor")
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    for name in ("lightllm.models", "lightllm.models.minicpmv4_6", "lightllm.models.qwen3_5"):
        mock_package(monkeypatch, name)
    mini = ModuleType("lightllm.models.minicpmv4_6.tokenizer")
    loop = ModuleType("lightllm.models.qwen3_5.model")
    mini.MiniCPMV4_6Tokenizer = lambda **kwargs: SimpleNamespace(kind="mini", **kwargs)
    loop.QWen3_5Tokenizer = lambda **kwargs: SimpleNamespace(kind="loop", **kwargs)
    monkeypatch.setitem(sys.modules, mini.__name__, mini)
    monkeypatch.setitem(sys.modules, loop.__name__, loop)

    result = namespace["get_tokenizer"]("synthetic-model")

    assert result.tokenizer is tokenizer
    assert result.model_cfg is config
    if model_type == "minicpmv4_6":
        assert result.kind == "mini" and result.image_processor == "mini_image"
        assert calls == ["AutoProcessor"]
    else:
        assert result.kind == "loop" and result.image_processor == "loop_image"
        assert calls == ["AutoImageProcessor"]
