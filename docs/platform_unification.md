# Multimodal inference on CUDA, MACA and MLU

The platform refactor provides a shared model and serving implementation with
`cuda`, `maca` and `mlu` backends. This change ports the existing LLaVA, InternVL
and MiniCPM-V-4.6 adaptations onto that interface.

## Runtime selection

Use the vendor environment for each device. PyTorch, Triton, collective libraries
and compiled extensions remain specific to that environment. Run the same source
checkout and select its backend with `--hardware_platform`.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# NVIDIA
CUDA_VISIBLE_DEVICES=0 python -m lightllm.server.api_server \
  --hardware_platform cuda --model_dir /path/to/model \
  --data_type fp16 --vit_att_backend sdpa

# MetaX C500
MACA_PATH=/opt/maca CUDA_VISIBLE_DEVICES=0 python -m lightllm.server.api_server \
  --hardware_platform maca --model_dir /path/to/model \
  --data_type fp16 --vit_att_backend sdpa

# Cambricon MLU
LD_LIBRARY_PATH=/usr/local/neuware/lib64:/usr/local/neuware/lib:$LD_LIBRARY_PATH \
MLU_VISIBLE_DEVICES=0 python -m lightllm.server.api_server \
  --hardware_platform mlu --model_dir /path/to/model \
  --data_type fp16 --vit_att_backend sdpa \
  --llm_prefill_att_backend sdpa --llm_decode_att_backend triton
```

Choose `--data_type` according to the checkpoint. The examples use the fp16
LLaVA-1.5 checkpoint; the tested MiniCPM-V-4.6 and InternVL checkpoints use bf16.
MLU registration is handled by `MluRuntime`; no `sitecustomize.py` or global
replacement of `Tensor.is_cuda` is required.

## Integration details

- Device placement checks use `BackendRuntime.is_device_tensor`, including KV
  movement, logprobs and speculative input handling.
- Graph inputs retain their storage when refreshed. MACA batches compatible
  copies through its runtime, while the other runtimes preserve individual
  nonblocking copies.
- MACA 3.8 and later register shared host memory as a whole region. Earlier
  runtime versions keep their existing segmented mapping implementation.
- LLaVA resolves missing Hugging Face configuration defaults, loads the current
  Transformers model layout and returns flattened image-token embeddings to the
  shared cache. Ordinary LLaVA checkpoints select the LLaVA model rather than the
  conditional Tarsier model registration.
- MiniCPM-V-4.6 reuses the Qwen3.5 text backbone and supplies its own image slicing,
  token expansion, visual encoder and continuous position state. Image segment
  order is shared by tokenization and visual embedding production.

## MLU attention and vision options

The registered `sdpa` text attention backend uses native SDPA for eager prefill,
including paged KV and cached prefixes. Decode uses the existing Triton path.
Sliding-window and graph-captured prefill use the Triton implementation. Prefill
metadata plans are scoped to one inference state and shared across its layers.

The existing optional LLaVA vision optimizations remain available on MLU:

- `LIGHTLLM_VIT_MLUGRAPH=0` disables vision graph caching.
- `LIGHTLLM_VIT_MLUGRAPH_CACHE` limits cached vision shapes (default 32).
- `LIGHTLLM_LLAVA_FAST_PREPROCESS=0` selects the Transformers image processor.
  The retained PIL/NumPy path is enabled by default on MLU; it can differ slightly
  in interpolation rounding. CUDA and MACA retain the Transformers path by default.

## Regression checks

Run these with the selected vendor environment and the checkout on `PYTHONPATH`:

```bash
python -m pytest -q \
  unit_tests/models/test_llava_tokenizer.py \
  unit_tests/models/test_minicpmv46_image_layout.py \
  unit_tests/models/test_draft_registry_target.py \
  unit_tests/platform/test_runtime_tensor_copy.py \
  unit_tests/platform/test_mlu_sdpa_prefill.py \
  unit_tests/utils/test_maca_host_registration.py \
  unit_tests/common/basemodel/triton_kernel/linear_att/test_fused_recurrent_strided.py
```

On 2026-10-29, these checks passed with 55 tests on A100, 55 on C500, and 51 on
MLU. The CUDA-specific foreach test and GDN kernel test module were skipped on
MLU. They cover image layout
against the Transformers processor, storage-preserving copies, causal prefill
with paged KV and cached prefixes, version-dependent host registration, and
native device-graph replay after refreshing captured inputs, and strided GDN
decode inputs on CUDA-compatible devices.

The API smoke runner generates solid-color images and exercises text, one image,
two ordered images, repeated-image caching and a large sliced image:

```bash
python test/test_api/multimodal_platform_smoke.py --base-url http://127.0.0.1:8000
```

LLaVA-1.5-7B, InternVL3.5-4B and MiniCPM-V-4.6 passed all five requests on each
of A100, C500 and MLU590 with single-device inference, SDPA vision attention and
text decode graphs disabled. MLU used SDPA text prefill; CUDA and MACA used
Triton prefill. The MiniCPM checks used 16x downsampling and a bounded linear
attention cache for functional validation.
Generated fixture checks establish API integration, not benchmark accuracy.
These checks do not establish support for every model, quantization format,
distributed topology or PD/DSpark configuration.
