#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/data/zhangtaoshan/models/Qwen3.5-27B-DSpark}"
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-${MODEL_DIR}/dspark}"
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CUCC_PATH="${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}"
# This machine's existing cu-bridge build was installed separately.
if [[ ! -d "$CUCC_PATH" && -d /data/zhangtaoshan/dspark_integration_backup/cu-bridge ]]; then
    export CUCC_PATH=/data/zhangtaoshan/dspark_integration_backup/cu-bridge
fi
export CUDA_PATH="$CUCC_PATH" CUCC_CMAKE_ENTRY=2
export PATH="$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$CUCC_PATH/tools:$CUCC_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/ompi/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1 TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1 MACA_QUEUE_SCHEDULE_POLICY=1
export MACA_GRAPH_LAUNCH_MODE="${MACA_GRAPH_LAUNCH_MODE:-1}" MACA_DIRECT_DISPATCH="${MACA_DIRECT_DISPATCH:-1}"
export MACA_TORCH_COMPILE_CONF="${MACA_TORCH_COMPILE_CONF:-triton.multi_kernel:1}"
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL="${LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-0}" LOADWORKER="${LOADWORKER:-16}"
export PAGE_SIZE="${PAGE_SIZE:-128}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
cd "$ROOT"
# mtp_* is LightLLM's shared speculative-decoding CLI. The selected implementation
# is DSpark, with the independent dspark/ checkpoint; dynamic verify is disabled.
exec python -m lightllm.server.api_server \
    --tp "${TP:-4}" --model_dir "$MODEL_DIR" --hardware_platform maca \
    --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 --data_type bfloat16 \
    --max_req_total_len "${MAX_REQ_TOTAL_LEN:-8192}" --max_total_token_num "${MAX_TOTAL_TOKEN_NUM:-32768}" \
    --batch_max_tokens "${BATCH_MAX_TOKENS:-4096}" --chunked_prefill_size "${CHUNKED_PREFILL_SIZE:-2048}" \
    --running_max_req_size "${RUNNING_MAX_REQ_SIZE:-8}" --graph_max_batch_size "${GRAPH_MAX_BATCH_SIZE:-4}" \
    --graph_max_len_in_batch "${GRAPH_MAX_LEN_IN_BATCH:-8192}" \
    --disable_aggressive_schedule --router_max_wait_tokens 600 --disable_vision --disable_audio \
    --use_tgi_api --trust_remote_code --host "${HOST:-127.0.0.1}" --port "${PORT:-19006}" \
    --mtp_mode dspark --mtp_draft_model_dir "$DRAFT_MODEL_DIR" --mtp_step "${DRAFT_STEP:-5}"
