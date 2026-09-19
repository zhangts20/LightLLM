import torch

from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_forward
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.qwen3_dflash.infer_struct import Qwen3DFlashInferStateInfo
from lightllm.models.qwen3_dflash.layer_weights.transformer_layer_weight import Qwen3DFlashTransformerLayerWeight
from lightllm.utils.envs_utils import get_env_start_args


class Qwen3DFlashTransformerLayerInfer(LlamaTransformerLayerInfer):
    """DFlash layer inference.

    The model path is built from two explicit layer primitives:
    - commit accepted target hidden rows into draft KV
    - run one non-causal draft block over prefix KV + scratch KV
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]
        self.partial_rotary_factor = network_config.get("partial_rotary_factor", 1.0)
        rope_scaling = network_config.get("rope_scaling") or {}
        mrope_section = rope_scaling.get("mrope_section")
        self.use_mrope = (
            bool(mrope_section)
            and getattr(get_env_start_args(), "hardware_platform", "cuda") == "ascend"
        )
        self.mrope_section = (
            torch.tensor(mrope_section, dtype=torch.int32, device=self.target_device)
            if self.use_mrope
            else None
        )

    def _apply_rotary(self, q, k, infer_state: Qwen3DFlashInferStateInfo):
        if self.use_mrope and infer_state.position_cos is not None and infer_state.position_cos.ndim == 3:
            if q is None:
                q = torch.empty_like(k)
            mrope_triton_fused(
                q,
                k,
                infer_state.position_cos,
                infer_state.position_sin,
                self.mrope_section,
                is_interleaved=True,
                partial_rotary_factor=self.partial_rotary_factor,
            )
            return
        rotary_emb_fwd(
            q if q is not None else k,
            k if q is not None else None,
            infer_state.position_cos,
            infer_state.position_sin,
            partial_rotary_factor=self.partial_rotary_factor,
        )

    def context_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state: Qwen3DFlashInferStateInfo,
        layer_weight: Qwen3DFlashTransformerLayerWeight,
    ) -> torch.Tensor:
        token_num, _ = input_embdings.shape
        cache_kv = layer_weight.kv_proj.mm(input_embdings, use_custom_tensor_mananger=False)
        qk_rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            layer_weight.qk_norm_weight_.k_weight,
            self.eps_,
        )
        cache_kv = cache_kv.view(token_num, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        self._apply_rotary(None, cache_kv[:, : self.tp_k_head_num_, :], infer_state)
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        return input_embdings

    def _get_qkv(self, input, infer_state: Qwen3DFlashInferStateInfo, layer_weight: Qwen3DFlashTransformerLayerWeight):
        q = layer_weight.q_proj.mm(input, use_custom_tensor_mananger=False)
        cache_kv = layer_weight.kv_proj.mm(input, use_custom_tensor_mananger=False)

        layer_weight.qk_norm_weight_(
            q,
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(
            -1,
            self.tp_k_head_num_ + self.tp_v_head_num_,
            self.head_dim_,
        )

        self._apply_rotary(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state,
        )
        return q, cache_kv
