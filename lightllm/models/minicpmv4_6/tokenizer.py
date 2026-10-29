"""MiniCPM-V-4.6 的 tokenizer 包装。

chat template（chat_template.jinja 的 render_content）对每个图片内容项只渲染一个字面量
`<|image_pad|>`，真正的视觉占位结构由 `MiniCPMV4_6Processor` 在文本层展开，见
transformers/models/minicpmv4_6/processing_minicpmv4_6.py L104-137：

    <image_id>{k}</image_id><image>{pads_global}</image>
    <slice>{pads_slice}</slice>...        # 共 slice_rows 行，每行 slice_cols 个，行间 "\\n"

这里复刻同一套展开（保留 image_id、image/slice 边界与换行的位置），再把每一段
pad 换成 embed cache 分配到的伪 token id：

    <image>  {token_id, ..., token_id + global_num - 1}          </image>
    <slice>  {token_id + global_num, ...}                        </slice>
    ...

伪 id 在序列里不连续也不影响注入：`multimodal_emb` kernel 用 `token_id - start_token_id`
作为 embed cache 的行偏移（common/basemodel/triton_kernel/multimodal_emb.py L67），
只要伪 id 的分配顺序与 visual server 输出的 embeds 顺序（全局视图在前、切片按行优先）
一致即可。
"""

from typing import List, Union

from lightllm.common.basemodel.multimodal_tokenizer import BaseMultiModalTokenizer
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import AudioItem, MultimodalParams, ImageItem
from lightllm.utils.log_utils import init_logger

from .vision_process import (
    MAX_SLICE_NUMS,
    PATCH_SIZE,
    SCALE_RESOLUTION,
    TOKEN_DIVISOR,
    ImageLayout,
    compute_image_layout,
)

logger = init_logger(__name__)

IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
IMAGE_START_TOKEN = "<image>"
IMAGE_END_TOKEN = "</image>"
SLICE_START_TOKEN = "<slice>"
SLICE_END_TOKEN = "</slice>"
IMAGE_ID_START_TOKEN = "<image_id>"
IMAGE_ID_END_TOKEN = "</image_id>"


class MiniCPMV4_6Tokenizer(BaseMultiModalTokenizer):
    def __init__(self, tokenizer=None, image_processor=None, model_cfg=None, **kwargs):
        super().__init__(tokenizer)
        self.image_processor = image_processor
        # 切片参数一律取自 AutoProcessor 里的 image_processor，保证与 visual server
        # 侧真实预处理完全一致（preprocessor_config.json 的字段可被覆盖）。
        self.max_slice_nums = int(getattr(image_processor, "max_slice_nums", MAX_SLICE_NUMS))
        self.scale_resolution = int(getattr(image_processor, "scale_resolution", SCALE_RESOLUTION))
        self.patch_size = int(getattr(image_processor, "patch_size", PATCH_SIZE))
        self.slice_mode = bool(getattr(image_processor, "slice_mode", True))
        self.use_image_id = bool(getattr(image_processor, "use_image_id", True))
        downsample_mode = getattr(image_processor, "downsample_mode", "16x") or "16x"
        self.token_divisor = 4 if downsample_mode == "4x" else TOKEN_DIVISOR

        self.image_token = IMAGE_TOKEN
        self.video_token = VIDEO_TOKEN
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def init_imageitem_extral_params(
        self, img: ImageItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        # 视觉 token 数只由图片宽高决定，不需要额外参数
        return

    def init_audioitem_extral_params(
        self, audio: AudioItem, multi_params: MultimodalParams, sampling_params: SamplingParams
    ):
        raise NotImplementedError

    def get_audio_token_length(self, audio: AudioItem):
        raise NotImplementedError

    def get_image_token_length(self, img: ImageItem):
        return self._layout(img).total_token_num

    def encode(self, prompt: Union[str, List[int]], multimodal_params: MultimodalParams = None, **kwargs):
        add_special_tokens = kwargs.get("add_special_tokens", True)
        if multimodal_params is not None and len(multimodal_params.audios) > 0:
            raise NotImplementedError("MiniCPM-V-4.6 audio input is not supported by lightllm yet.")
        images = [] if multimodal_params is None else list(multimodal_params.images)

        if len(images) == 0:
            return self._plain_encode(prompt, add_special_tokens)

        layouts = [self._layout(img) for img in images]
        if isinstance(prompt, str):
            return self._encode_text(prompt, images, layouts, add_special_tokens)
        if isinstance(prompt, list):
            return self._encode_ids(prompt, images, layouts)
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    # ------------------------------------------------------------------ helpers

    def _layout(self, img: ImageItem) -> ImageLayout:
        return compute_image_layout(
            width=img.image_w,
            height=img.image_h,
            scale_resolution=self.scale_resolution,
            patch_size=self.patch_size,
            max_slice_nums=self.max_slice_nums,
            slice_mode=self.slice_mode,
            token_divisor=self.token_divisor,
        )

    def _plain_encode(self, prompt, add_special_tokens: bool):
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        if isinstance(prompt, list):
            return list(prompt)
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    @staticmethod
    def _segment_token_nums(layout: ImageLayout) -> List[int]:
        """一张图各 NaViT 段的视觉 token 数：全局视图 + 各切片（顺序与 target_sizes 一致）。"""
        return [layout.global_token_num] + [layout.slice_token_num] * layout.slice_num

    def _image_placeholder_text(self, image_index: int, layout: ImageLayout) -> str:
        placeholder = IMAGE_START_TOKEN + self.image_token * layout.global_token_num + IMAGE_END_TOKEN
        if self.use_image_id:
            placeholder = f"{IMAGE_ID_START_TOKEN}{image_index}{IMAGE_ID_END_TOKEN}" + placeholder
        if self.slice_mode and layout.slice_num > 0:
            slice_placeholder = SLICE_START_TOKEN + self.image_token * layout.slice_token_num + SLICE_END_TOKEN
            rows = [slice_placeholder * layout.slice_cols for _ in range(layout.slice_rows)]
            placeholder += "\n".join(rows)
        return placeholder

    def _expand_image_placeholders(self, chunks: List[str], layouts: List[ImageLayout]) -> str:
        # 注意不能用 str.replace(..., 1) 逐图替换：前一张图展开出来的占位文本里也含有
        # `<|image_pad|>`，会被后续 replace 命中。按占位符把 prompt 切开再拼接才安全。
        parts = [chunks[0]]
        for image_index, layout in enumerate(layouts):
            parts.append(self._image_placeholder_text(image_index, layout))
            parts.append(chunks[image_index + 1])
        return "".join(parts)

    @staticmethod
    def _pad_runs(ids: List[int], pad_id: int):
        runs = []
        i, n = 0, len(ids)
        while i < n:
            if ids[i] == pad_id:
                j = i
                while j < n and ids[j] == pad_id:
                    j += 1
                runs.append((i, j - i))
                i = j
            else:
                i += 1
        return runs

    def _image_segment_ids(self, image_index: int, layout: ImageLayout, token_id: int) -> List[int]:
        """预分词（List[int]）输入时用：直接拼出该图的完整 token 结构。"""
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        ids = []
        if self.use_image_id:
            ids.extend(encode(f"{IMAGE_ID_START_TOKEN}{image_index}{IMAGE_ID_END_TOKEN}"))
        ids.extend(encode(IMAGE_START_TOKEN))
        offset = layout.global_token_num
        ids.extend(range(token_id, token_id + offset))
        ids.extend(encode(IMAGE_END_TOKEN))
        if self.slice_mode and layout.slice_num > 0:
            slice_start_ids = encode(SLICE_START_TOKEN)
            slice_end_ids = encode(SLICE_END_TOKEN)
            newline_ids = encode("\n")
            for row in range(layout.slice_rows):
                for _ in range(layout.slice_cols):
                    ids.extend(slice_start_ids)
                    ids.extend(range(token_id + offset, token_id + offset + layout.slice_token_num))
                    ids.extend(slice_end_ids)
                    offset += layout.slice_token_num
                if row != layout.slice_rows - 1:
                    ids.extend(newline_ids)
        return ids

    def _encode_text(self, prompt: str, images, layouts, add_special_tokens: bool) -> List[int]:
        if self.video_token in prompt:
            raise NotImplementedError("MiniCPM-V-4.6 video input is not supported by lightllm yet.")
        chunks = prompt.split(self.image_token)
        if len(chunks) != len(images) + 1:
            raise ValueError(
                f"image placeholder num {len(chunks) - 1} != image num {len(images)}, "
                f"the prompt and multimodal_params are inconsistent."
            )

        expanded = self._expand_image_placeholders(chunks, layouts)
        origin_ids = self.tokenizer.encode(expanded, add_special_tokens=add_special_tokens)

        expected = []
        for layout in layouts:
            expected.extend(self._segment_token_nums(layout))
        runs = self._pad_runs(origin_ids, self.image_token_id)
        if len(runs) != len(expected):
            raise ValueError(
                f"image pad segment num mismatch: prompt has {len(runs)} runs, expected {len(expected)} "
                f"for {len(images)} image(s), check the image processor config."
            )

        input_ids = []
        cursor = 0
        run_index = 0
        for img, layout in zip(images, layouts):
            if img.token_id is None:
                raise ValueError("image token_id must be allocated before tokenizer.encode.")
            offset = 0
            for seg_len in self._segment_token_nums(layout):
                pos, run_len = runs[run_index]
                run_index += 1
                if run_len != seg_len:
                    raise ValueError(
                        f"image pad run length {run_len} != expected {seg_len} " f"(image {img.image_w}x{img.image_h})."
                    )
                input_ids.extend(origin_ids[cursor:pos])
                if img.start_idx is None:
                    img.start_idx = len(input_ids)
                input_ids.extend(range(img.token_id + offset, img.token_id + offset + seg_len))
                cursor = pos + run_len
                offset += seg_len
        input_ids.extend(origin_ids[cursor:])
        return input_ids

    def _encode_ids(self, prompt_ids: List[int], images, layouts) -> List[int]:
        positions = [i for i, token_id in enumerate(prompt_ids) if token_id == self.image_token_id]
        if len(positions) != len(images):
            raise ValueError(
                f"image placeholder num {len(positions)} != image num {len(images)}, "
                f"the prompt and multimodal_params are inconsistent."
            )

        input_ids = []
        cursor = 0
        for image_index, (img, layout, pos) in enumerate(zip(images, layouts, positions)):
            if img.token_id is None:
                raise ValueError("image token_id must be allocated before tokenizer.encode.")
            input_ids.extend(prompt_ids[cursor:pos])
            img.start_idx = len(input_ids)
            input_ids.extend(self._image_segment_ids(image_index, layout, img.token_id))
            cursor = pos + 1
        input_ids.extend(prompt_ids[cursor:])
        return input_ids
