"""MiniCPM-V-4.6 图像切片/布局数学。

这里是 transformers `MiniCPMV4_6ImageProcessorPil` 中切片逻辑的等价复刻，用于在
tokenizer 侧（只有图片宽高，没有像素）推算一张图会占多少个 `<|image_pad|>` token。

上游对应实现见
transformers/models/minicpmv4_6/image_processing_pil_minicpmv4_6.py：
    ensure_divide / find_best_resize / get_refine_size / get_sliced_grid / _preprocess
"""

import math

# 与 preprocessor_config.json 一致
SCALE_RESOLUTION = 448
PATCH_SIZE = 14
MAX_SLICE_NUMS = 9
SLICE_MODE = True
USE_IMAGE_ID = True

# "16x" 下采样模式：vit_merger(2x2) + merger MLP(2x2) 一共把 patch 数缩减 16 倍
TOKEN_DIVISOR = 16
# 每次 2x2 合并要求边长是 patch_size * 2 的整数倍，两次合并即 patch_size * 4
RESIZE_FACTOR = PATCH_SIZE * 4


def ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def find_best_resize(image_size, scale_resolution: int, patch_size: int, allow_upscale: bool = False):
    height, width = image_size
    if (height * width > scale_resolution * scale_resolution) or allow_upscale:
        aspect_ratio = width / height
        height = int(scale_resolution / math.sqrt(aspect_ratio))
        width = int(height * aspect_ratio)
    best_width = ensure_divide(width, patch_size * 4)
    best_height = ensure_divide(height, patch_size * 4)
    return best_height, best_width


def get_refine_size(image_size, grid, scale_resolution: int, patch_size: int, allow_upscale: bool = False):
    height, width = image_size
    grid_y, grid_x = grid
    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    best_height, best_width = find_best_resize(
        image_size=(refine_height / grid_y, refine_width / grid_x),
        scale_resolution=scale_resolution,
        patch_size=patch_size,
        allow_upscale=allow_upscale,
    )
    return best_height * grid_y, best_width * grid_x


def get_sliced_grid(image_size, max_slice_nums: int, scale_resolution: int):
    original_height, original_width = image_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)
    if multiple <= 1:
        return None

    best_grid = [1, 1]
    min_error = float("inf")
    for num_slices in [multiple - 1, multiple, multiple + 1]:
        if num_slices == 1 or num_slices > max_slice_nums:
            continue
        for num_rows in range(1, num_slices + 1):
            if num_slices % num_rows == 0:
                num_cols = num_slices // num_rows
                error = abs(log_ratio - math.log(num_rows / num_cols))
                if error < min_error:
                    best_grid = [num_cols, num_rows]
                    min_error = error
    return best_grid


class ImageLayout:
    """一张图在 16x 下采样模式下的视觉 token 布局。

    属性中的 patches 均为 ViT 输入侧的 patch 网格 (h, w)，即像素尺寸除以 patch_size。
    全局视图与切片视图各自是一个独立的 NaViT 段。
    """

    def __init__(self, global_patches, slice_patches, slice_rows, slice_cols, token_divisor=TOKEN_DIVISOR):
        self.global_patches = global_patches
        self.slice_patches = slice_patches
        self.slice_rows = slice_rows
        self.slice_cols = slice_cols
        self.token_divisor = token_divisor

    @property
    def global_token_num(self) -> int:
        h, w = self.global_patches
        return (h * w) // self.token_divisor

    @property
    def slice_token_num(self) -> int:
        if self.slice_patches is None:
            return 0
        h, w = self.slice_patches
        return (h * w) // self.token_divisor

    @property
    def slice_num(self) -> int:
        if self.slice_patches is None:
            return 0
        return self.slice_rows * self.slice_cols

    @property
    def total_token_num(self) -> int:
        return self.global_token_num + self.slice_num * self.slice_token_num


def compute_image_layout(
    width: int,
    height: int,
    scale_resolution: int = SCALE_RESOLUTION,
    patch_size: int = PATCH_SIZE,
    max_slice_nums: int = MAX_SLICE_NUMS,
    slice_mode: bool = SLICE_MODE,
    token_divisor: int = TOKEN_DIVISOR,
) -> ImageLayout:
    """复刻 MiniCPMV4_6Processor/image_processor 的已切图逻辑，得到 token 布局。"""
    image_size = (height, width)
    best_grid = get_sliced_grid(image_size, max_slice_nums, scale_resolution) if slice_mode else None

    source_height, source_width = find_best_resize(
        image_size, scale_resolution, patch_size, allow_upscale=(best_grid is None)
    )
    global_patches = (source_height // patch_size, source_width // patch_size)

    slice_patches = None
    slice_rows = slice_cols = 0
    if best_grid is not None:
        refine_height, refine_width = get_refine_size(
            image_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )
        grid_y, grid_x = best_grid
        # divide_to_patches 按 (refine_height // grid_y, refine_width // grid_x) 切分，
        # 得到的 tile 网格是 grid_y 行 × grid_x 列
        patch_height = refine_height // grid_y
        patch_width = refine_width // grid_x
        slice_patches = (patch_height // patch_size, patch_width // patch_size)
        slice_rows, slice_cols = grid_y, grid_x

    return ImageLayout(global_patches, slice_patches, slice_rows, slice_cols, token_divisor)
