"""Target-aware adapter selection without loading model weights or accelerators."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("draft_registry_under_test", ROOT / "lightllm/models/draft_registry.py")
registry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry_module)


@pytest.fixture
def registry(monkeypatch):
    registry = registry_module._DraftModelRegistry()
    for model_type in ("qwen3", "qwen3_5", "deepseek_v3", "qwen3_5_moe"):
        for mode in ("dspark", "dflash", "eagle3", "vanilla_with_att"):
            cls = type(f"{model_type}_{mode}", (), {})
            registry(model_type=model_type, spec_modes=mode)(cls)
    monkeypatch.setattr(registry_module, "DraftModelRegistry", registry)
    return registry


@pytest.mark.parametrize("mode", ["dspark", "dflash"])
@pytest.mark.parametrize("target_type", ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_qwen3_checkpoint_uses_qwen35_target_adapter(registry, mode, target_type, wrapped):
    target = {"model_type": target_type}
    if wrapped:
        target = {"model_type": "multimodal", "text_config": target}
    draft = {"model_type": "qwen3", "rope_parameters": {"partial_rotary_factor": .25}}
    selected = registry_module.get_draft_model_class(draft, mode, target_model_cfg=target)
    assert selected is registry._registry[("qwen3_5", mode)]
    assert draft == {"model_type": "qwen3", "rope_parameters": {"partial_rotary_factor": .25}}


@pytest.mark.parametrize("draft_type, mode, target", [
    ("qwen3", "dspark", None),
    ("qwen3", "dflash", {"model_type": "qwen3"}),
    ("qwen3", "eagle3", {"model_type": "qwen3_5"}),
    ("deepseek_v3", "vanilla_with_att", {"model_type": "qwen3_5"}),
    ("qwen3_5_moe", "vanilla_with_att", {"model_type": "qwen3_5_moe"}),
    ("qwen3_5", "dspark", {"model_type": "qwen3_5"}),
])
def test_other_registrations_unchanged(registry, draft_type, mode, target):
    assert registry_module.get_draft_model_class(
        {"model_type": draft_type}, mode, target_model_cfg=target
    ) is registry._registry[(draft_type, mode)]


def test_unknown_draft_still_errors(registry):
    with pytest.raises(ValueError, match="Unsupported speculative draft model"):
        registry_module.get_draft_model_class(
            {"model_type": "unknown"}, "dspark", target_model_cfg={"model_type": "qwen3_5"}
        )
