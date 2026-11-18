"""Check image token accounting against the Transformers processor."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "minicpmv46_layout", ROOT / "lightllm/models/minicpmv4_6/vision_process.py"
)
layout_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout_module)


@pytest.mark.parametrize("width,height", [(64, 32), (336, 336), (500, 320), (1600, 500), (500, 1600), (1920, 1080)])
@pytest.mark.parametrize("slice_mode", [False, True])
def test_layout_matches_transformers_target_sizes(width, height, slice_mode):
    module = pytest.importorskip("transformers.models.minicpmv4_6.image_processing_pil_minicpmv4_6")
    from PIL import Image

    processor = module.MiniCPMV4_6ImageProcessorPil(
        scale_resolution=448, patch_size=14, max_slice_nums=9, slice_mode=slice_mode, downsample_mode="16x"
    )
    target_sizes = processor(Image.new("RGB", (width, height)), return_tensors="pt")["target_sizes"]
    layout = layout_module.compute_image_layout(width, height, slice_mode=slice_mode)
    assert tuple(target_sizes[0].tolist()) == layout.global_patches
    assert target_sizes.shape[0] == 1 + layout.slice_num
    assert layout.total_token_num == int(target_sizes.prod(-1).sum()) // 16
    if layout.slice_num:
        assert all(tuple(size.tolist()) == layout.slice_patches for size in target_sizes[1:])


def test_small_square_keeps_one_global_view():
    layout = layout_module.compute_image_layout(224, 224)
    assert layout.global_patches == (32, 32)
    assert layout.total_token_num == 64
    assert layout.slice_num == 0
