import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from PIL import Image
from typing import List, Union
from safetensors import safe_open
from io import BytesIO
from lightllm.models.visual_utils import VisualDeviceMixin
from lightllm.platform import get_backend
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class _VisionGraphRunner:
    def __init__(self, model, pixel_values):
        self.backend = get_backend()
        self.input = torch.zeros_like(pixel_values)
        with torch.no_grad():
            for _ in range(2):
                model._vit_forward_impl(self.input)
            self.backend.runtime.synchronize()
            self.pool = self.backend.graph.graph_pool_handle()
            self.graph = self.backend.graph.create_graph()
            with self.backend.graph.graph(self.graph, pool=self.pool):
                self.output = model._vit_forward_impl(self.input)
        self.backend.graph.replay_graph(self.graph)
        self.backend.runtime.synchronize()

    def run(self, pixel_values):
        self.input.copy_(pixel_values)
        self.backend.graph.replay_graph(self.graph)
        return self.output


class LlavaVisionModel(VisualDeviceMixin):
    def __init__(self):
        self._vit_runners = {}
        self._vit_runner_failed = set()

    def _device_module_attrs(self):
        return ("vision_tower",)

    def _device_tensor_dict_attrs(self):
        return ("projector_weights",)

    def load_model(self, weight_dir):
        self._vit_runners.clear()
        self._vit_runner_failed.clear()
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))

        # for llava-v1.5-7b-hf model, should load config from transformers
        if "text_config" in config:
            self.load_hf_model(config, weight_dir)
        else:
            self.load_bin_model(config, weight_dir)

        self.vision_tower.requires_grad_(False)

        assert "model.mm_projector.0.weight" in self.projector_weights
        assert "model.mm_projector.0.bias" in self.projector_weights
        assert "model.mm_projector.2.weight" in self.projector_weights
        assert "model.mm_projector.2.bias" in self.projector_weights

    def load_hf_model(self, config, weight_dir):
        from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

        config = AutoConfig.from_pretrained(weight_dir, trust_remote_code=True)
        self.select_layer = config.vision_feature_layer
        self.select_feature = config.vision_feature_select_strategy
        processor = AutoProcessor.from_pretrained(weight_dir)
        self.image_processor = processor.image_processor
        model = LlavaForConditionalGeneration.from_pretrained(
            weight_dir,
            torch_dtype=torch.float16,
        )
        base = getattr(model, "model", model)
        self.vision_tower = base.vision_tower
        base.multi_modal_projector = None
        base.language_model = None

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".safetensors"):
                d = safe_open(os.path.join(weight_dir, f), "pt", "cpu")
                for k in d.keys():
                    if "multi_modal_projector.linear_1" in k:
                        self.projector_weights[
                            "model.mm_projector.0" + k.split("multi_modal_projector.linear_1", 1)[1]
                        ] = d.get_tensor(k).half()
                    if "multi_modal_projector.linear_2" in k:
                        self.projector_weights[
                            "model.mm_projector.2" + k.split("multi_modal_projector.linear_2", 1)[1]
                        ] = d.get_tensor(k).half()

    def load_bin_model(self, config, weight_dir):
        self.select_layer = config.get("mm_vision_select_layer", -2)
        self.select_feature = config.get("mm_vision_select_feature", "patch")

        # load clip vision model by cfg['mm_vision_tower']:
        #   huggingface_name or path_of_clip_relative_to_llava_model_dir
        vision_path = config.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
        if isinstance(vision_path, list):
            vision_path = vision_path[0]
        if vision_path.startswith("./"):
            vision_path = os.path.join(weight_dir, vision_path)

        from transformers import CLIPVisionModel, CLIPImageProcessor

        self.image_processor = CLIPImageProcessor.from_pretrained(vision_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_path).half()

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".bin"):
                d = torch.load(os.path.join(weight_dir, f), "cpu")
                for k, v in d.items():
                    if "model.mm_projector" in k:
                        self.projector_weights[k] = v.half()

    # batch images infer
    def forward(self, x):
        x = self.move_to_infer_device(x.half())
        if self.infer_device.type == "mlu" and os.getenv("LIGHTLLM_VIT_MLUGRAPH", "1") == "1":
            key = (x.device, x.dtype, tuple(x.shape))
            runner = self._vit_runners.get(key)
            if runner is None and key not in self._vit_runner_failed:
                capacity = int(os.getenv("LIGHTLLM_VIT_MLUGRAPH_CACHE", "32"))
                if len(self._vit_runners) < capacity:
                    try:
                        runner = _VisionGraphRunner(self, x)
                        self._vit_runners[key] = runner
                    except RuntimeError as error:
                        logger.warning("Vision graph capture failed for %s; using eager: %s", key, error)
                        self._vit_runner_failed.add(key)
                else:
                    self._vit_runner_failed.add(key)
            if runner is not None:
                return runner.run(x)
        return self._vit_forward_impl(x)

    def _vit_forward_impl(self, x):
        x = self.vision_tower(x, output_hidden_states=True)
        x = x.hidden_states[self.select_layer]
        if self.select_feature == "patch" or self.select_feature == "default":
            x = x[:, 1:].contiguous()
        B, L, N = x.shape
        x = x.view(-1, N).half()

        # mm_project
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.0.weight"],
            bias=self.projector_weights["model.mm_projector.0.bias"],
        )
        x = F.gelu(x)
        x = F.linear(
            x,
            weight=self.projector_weights["model.mm_projector.2.weight"],
            bias=self.projector_weights["model.mm_projector.2.bias"],
        )
        x = x.view(B, L, -1)
        return x

    @staticmethod
    def _processor_field(config, key):
        return config.get(key) if isinstance(config, dict) else getattr(config, key, None)

    def _preprocess_image(self, image):
        """Retain the existing optional PIL/NumPy CLIP preprocessing path."""
        processor = self.image_processor
        size = processor.size
        shortest_edge = self._processor_field(size, "shortest_edge")
        width, height = image.size
        if shortest_edge:
            if width <= height:
                new_width, new_height = int(shortest_edge), int(shortest_edge * height / width)
            else:
                new_height, new_width = int(shortest_edge), int(shortest_edge * width / height)
        else:
            new_height = int(self._processor_field(size, "height"))
            new_width = int(self._processor_field(size, "width"))
        if (new_width, new_height) != image.size:
            image = image.resize((new_width, new_height), processor.resample)
        if getattr(processor, "do_center_crop", False):
            crop_height = int(self._processor_field(processor.crop_size, "height"))
            crop_width = int(self._processor_field(processor.crop_size, "width"))
            left, top = (new_width - crop_width) // 2, (new_height - crop_height) // 2
            image = image.crop((left, top, left + crop_width, top + crop_height))
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) * processor.rescale_factor
        pixels = pixels.transpose(2, 0, 1)
        mean = np.asarray(processor.image_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(processor.image_std, dtype=np.float32).reshape(3, 1, 1)
        return torch.from_numpy((pixels - mean) / std)

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        uuids = []

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data)).convert("RGB")
                default_fast = "1" if self.infer_device.type == "mlu" else "0"
                if os.getenv("LIGHTLLM_LLAVA_FAST_PREPROCESS", default_fast) == "1":
                    t = self._preprocess_image(image_data).unsqueeze(0)
                else:
                    t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]
                img_tensors.append(t)
            else:
                raise Exception("Unsupported input types: {} for {}".format(type(img), img))

        if len(img_tensors) <= 0:
            return None

        img = torch.cat(img_tensors, dim=0)
        all_img_embeds = self.forward(img)
        batch_size, image_tokens, hidden_size = all_img_embeds.shape
        valid_ids = [[i * image_tokens, (i + 1) * image_tokens] for i in range(batch_size)]
        return all_img_embeds.reshape(-1, hidden_size), uuids, valid_ids
