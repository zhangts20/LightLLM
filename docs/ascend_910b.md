# Ascend 910B

Use an Ascend environment containing compatible CANN, PyTorch, TorchNPU,
torchvision and Triton-Ascend builds. The CUDA requirements file installs CUDA
extensions and should not be used to prepare this environment. See the
[Triton-Ascend installation guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/installation_guide.md)
for the vendor stack.

## Install and run

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-ascend.txt
pip install --no-deps -e .
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

PAGE_SIZE=128 ASCEND_RT_VISIBLE_DEVICES=0 python -m lightllm.server.api_server \
  --hardware_platform ascend --model_dir /path/to/Qwen3-0.6B --port 8910 \
  --data_type bf16 --max_total_token_num 4096 --batch_max_tokens 512 \
  --running_max_req_size 8 --max_req_total_len 1024 --disable_cudagraph
```

`PAGE_SIZE=128` selects paged KV storage and lets `auto` resolve FA3 to the
Ascend fused inference attention backend. Keep it set for every process in the
service. Without it the shared default remains one token per page, preserving
the existing CUDA/MACA/MLU behavior. Explicit `fa3` also resolves to this
Ascend backend with paged storage. Its automatic validation uses TorchNPU and a
CPU attention reference for both causal prefill and GQA decode.

The example disables decode graphs for a small initial serving check. NPU graph
input copying and replay have a separate regression test. A small native graph
test alone does not establish model-level graph support for all shapes.

Ascend runtime registration happens when `AscendRuntime` is constructed; other
platforms do not need to install TorchNPU. Vision LayerNorm uses native PyTorch
on the NPU, and graph tensor aliases use TorchNPU's weak-reference API instead
of CuPy. Tensor-cache storage counts support both the older TorchNPU API and
the upstream PyTorch API used by newer TorchNPU releases.

## Validation

The adaptation was exercised on Ascend910B2C with Python 3.12.16, PyTorch
2.10.0+cpu, torch_npu 2.10.0.post7, Triton 3.6.0 / triton_ascend
3.6.0.post2 and CANN 9.2.1. The host driver reports 26.0.rc1.

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest -q unit_tests/platform/test_ascend.py
```

The native tests compare LayerNorm, embedding shard masking, non-contiguous GQA
Q/K normalization and attention against references, and check NPU tensor aliases
and storage counts. The shared runtime-copy suite also includes native NPU graph
capture and replay when `LIGHTLLM_START_ARGS` selects `ascend`.

The API smoke runner checks eight requests: repeated greedy output, a long
prompt crossing page boundaries, a verified prefix cache hit, three concurrent
requests and streaming token equivalence:

```bash
python test/test_api/ascend_smoke.py --base-url http://127.0.0.1:8910
```

Qwen3-0.6B passed these checks in bf16 with decode graphs disabled. The repeated
324-token prompt reused 256 cached tokens; streaming returned the same 16 token
IDs as ordinary generation. The prompt "The capital of France is" continued
with " Paris.".

The existing LLaVA tokenizer, MiniCPM-V image layout, draft registry, MLU SDPA,
MACA host registration and runtime-copy checks were run in the Ascend environment.
Together with the native tests, 61 checks passed and the CUDA foreach test
was skipped. These exercise shared logic; CUDA, MACA and MLU hardware were not available on
this host. Full multimodal checkpoints, quantized inference, multi-device HCCL,
PD separation and speculative decoding require their own integration validation.
