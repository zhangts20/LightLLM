import torch
import torch.nn.functional as F
import torch.nn as nn
import json
import os
from PIL import Image
from typing import List, Union
from safetensors import safe_open
from io import BytesIO
from lightllm.models.visual_utils import VisualDeviceMixin
from lightllm.server.multimodal_params import MultimodalParams, ImageItem
from lightllm.server.embed_cache.utils import read_shm, get_shm_name_data
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class Gemma3VisionModel(VisualDeviceMixin):

    def _device_module_attrs(self):
        return ("vision_tower", "avg_pool",)

    def _device_tensor_dict_attrs(self):
        return ("projector_weights",)

    def load_model(self, weight_dir):
        config_file = os.path.join(weight_dir, "config.json")
        config = json.load(open(config_file))

        # for llava-v1.5-7b-hf model, should load config from transformers
        if "text_config" in config:
            self.load_hf_model(config, weight_dir)
        else:
            assert False, "only hf format model is supported for Gemma3"

        self.mm_tokens_per_image = int(config["mm_tokens_per_image"])
        self.patches_per_image = int(config["vision_config"]["image_size"] // config["vision_config"]["patch_size"])
        self.tokens_per_side = int(self.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

        self.vision_tower.requires_grad_(False)

        assert "model.mm_projector.linear" in self.projector_weights
        assert "model.mm_projector.norm" in self.projector_weights

    @staticmethod
    def _force_eager_attention(module):
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = "eager"
        for child in module.children():
            Gemma3VisionModel._force_eager_attention(child)

    def load_hf_model(self, config, weight_dir):
        from transformers import AutoConfig, AutoProcessor, Gemma3ForConditionalGeneration

        # config = AutoConfig.from_pretrained(weight_dir, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(weight_dir)
        self.image_processor = processor.image_processor

        # Match server --data_type bfloat16. transformers 5 SigLIP defaults to SDPA,
        # and new PyTorch cuDNN SDPA can fail with "No valid execution plans built".
        # Old transformers used eager attention; keep that path.
        load_kwargs = {"torch_dtype": torch.bfloat16}
        try:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                weight_dir, attn_implementation="eager", **load_kwargs
            )
        except TypeError:
            model = Gemma3ForConditionalGeneration.from_pretrained(weight_dir, **load_kwargs)
        # transformers <5: vision_tower lives on Gemma3ForConditionalGeneration.
        # transformers 5+: it lives on the inner Gemma3Model (`model.model`).
        inner_model = model.model if hasattr(model, "model") and not hasattr(model, "vision_tower") else model
        self.vision_tower = inner_model.vision_tower
        self._force_eager_attention(self.vision_tower)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        # Free projector/LLM memory. New transformers uses read-only properties on the
        # wrapper; fall back to inner model.model (Gemma3Model) when setattr fails.
        try:
            inner_model.multi_modal_projector = None
            inner_model.language_model = None
        except AttributeError:
            pass

        # load projector weights
        self.projector_weights = {}
        for f in os.listdir(weight_dir):
            if f.endswith(".safetensors"):
                d = safe_open(os.path.join(weight_dir, f), "pt", "cpu")
                for k in d.keys():
                    if "multi_modal_projector.mm_input_projection_weight" in k:
                        self.projector_weights[
                            k.replace("multi_modal_projector.mm_input_projection_weight", "model.mm_projector.linear")
                        ] = d.get_tensor(k).to(torch.bfloat16)
                    if "multi_modal_projector.mm_soft_emb_norm.weight" in k:
                        self.projector_weights[
                            k.replace("multi_modal_projector.mm_soft_emb_norm.weight", "model.mm_projector.norm")
                        ] = d.get_tensor(k).to(torch.bfloat16)

    def gemma3_rms_norm(self, input, weight, eps: float = 1e-6):
        def _norm(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(input.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + weight.float())
        return output.type_as(input)

    # batch images infer
    def forward(self, x):
        x = self.move_to_infer_device(x.to(torch.bfloat16))
        x = self.vision_tower(x, output_hidden_states=True).last_hidden_state

        batch_size, _, seq_length = x.shape

        reshaped_vision_outputs = x.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.gemma3_rms_norm(
            pooled_vision_outputs.float(), self.projector_weights["model.mm_projector.norm"]
        ).to(torch.bfloat16)

        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.projector_weights["model.mm_projector.linear"]
        )

        return projected_vision_outputs.type_as(x)

    def encode(self, images: List[ImageItem]):
        img_tensors = []
        uuids = []
        valid_id = 0
        valid_ids = []

        for i, img in enumerate(images):
            if isinstance(img, ImageItem):
                uuids.append(img.uuid)
                image_data = read_shm(get_shm_name_data(img.uuid))
                image_data = Image.open(BytesIO(image_data))
                t = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"]
                img_tensors.append(t)
            else:
                raise Exception("Unsupported input types: {} for {}".format(type(img), img))

            cur_num = img_tensors[-1].shape[0] * self.mm_tokens_per_image
            valid_ids.append([valid_id, valid_id + cur_num])
            valid_id += cur_num

        if len(img_tensors) <= 0:
            return None

        img = torch.cat(img_tensors, dim=0)
        all_img_embeds = self.forward(img)
        all_img_embeds = all_img_embeds.reshape(-1, all_img_embeds.shape[-1])

        return all_img_embeds, uuids, valid_ids
