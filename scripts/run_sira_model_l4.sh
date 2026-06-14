#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/content/Self-information-Rewrite-Attack"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/sira_outputs}"
ALGORITHM="${ALGORITHM:-KGW}"
SAMPLES="${SAMPLES:-500}"
GENERATION_MODEL="${GENERATION_MODEL:-facebook/opt-1.3b}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME before running this script}"
MODEL_LABEL="${MODEL_LABEL:?Set MODEL_LABEL before running this script}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOADER_TYPE="${LOADER_TYPE:-auto}"
GENERATED_TOKENS="${GENERATED_TOKENS:-230}"
BATCH_SIZE="${BATCH_SIZE:-4}"

WATERMARK_DIR="${OUTPUT_ROOT}/watermarked"
MODEL_DIR="${OUTPUT_ROOT}/sira_models/${MODEL_LABEL}"
INTERMEDIATE_DIR="${MODEL_DIR}/intermediate"
FINAL_DIR="${MODEL_DIR}/final"
LOG_FILE="${OUTPUT_ROOT}/logs/sira_${MODEL_LABEL}.log"

mkdir -p "${WATERMARK_DIR}" "${INTERMEDIATE_DIR}" "${FINAL_DIR}" "${OUTPUT_ROOT}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "[ERROR] SIRA model run failed at line ${LINENO} with exit status ${status}. Log: ${LOG_FILE}"; exit ${status}' ERR

QUANTIZATION_ARGS=()
if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
  QUANTIZATION_ARGS+=(--load_in_4bit)
fi

echo "Running SIRA with ${MODEL_NAME}"
echo "Label=${MODEL_LABEL}, algorithm=${ALGORITHM}, samples=${SAMPLES}, batch_size=${BATCH_SIZE}, load_in_4bit=${LOAD_IN_4BIT}, loader_type=${LOADER_TYPE}"

echo "=== Stage 0: Generate shared watermarked data ==="
WATERMARK_FILE="${WATERMARK_DIR}/${ALGORITHM}_response.json"
WATERMARK_COUNT=0
if [[ -f "${WATERMARK_FILE}" ]]; then
  WATERMARK_COUNT="$(wc -l < "${WATERMARK_FILE}")"
fi

if (( WATERMARK_COUNT >= SAMPLES )); then
  echo "Reusing ${WATERMARK_COUNT} completed watermarked samples."
else
  echo "Resuming watermarked data after ${WATERMARK_COUNT} samples."
  python scripts/generate_watermark.py \
    --algorithms "${ALGORITHM}" \
    --model_path "${GENERATION_MODEL}" \
    --input_path "${REPO_DIR}/dataset/c4/trim_c4_500.json" \
    --output_dir "${WATERMARK_DIR}" \
    --cuda_visible_devices 0 \
    --cuda_device 0 \
    --dtype bf16 \
    --max_samples "${SAMPLES}" \
    --max_new_tokens "${GENERATED_TOKENS}" \
    --min_length "${GENERATED_TOKENS}" \
    --seed 42
fi

echo "=== Stage 1 and 2: Paraphrase and self-information blanking ==="
python scripts/pre_attack.py \
  --input_dir "${WATERMARK_DIR}" \
  --result_dir "${INTERMEDIATE_DIR}" \
  --model_path "${MODEL_NAME}" \
  --threshold 30 \
  --gpu 0 \
  --algorithms "${ALGORITHM}" \
  --dtype bf16 \
  --loader_type "${LOADER_TYPE}" \
  --batch_size "${BATCH_SIZE}" \
  "${QUANTIZATION_ARGS[@]}" \
  --max_samples "${SAMPLES}" \
  --seed 42

echo "=== Stage 3: Rewrite attack ==="
python scripts/attack.py \
  --model_path "${MODEL_NAME}" \
  --input_dir "${INTERMEDIATE_DIR}" \
  --output_dir "${FINAL_DIR}" \
  --gpu 0 \
  --algorithms "${ALGORITHM}" \
  --dtype bf16 \
  --loader_type "${LOADER_TYPE}" \
  --batch_size "${BATCH_SIZE}" \
  "${QUANTIZATION_ARGS[@]}" \
  --max_samples "${SAMPLES}" \
  --seed 42

echo "SIRA output: ${FINAL_DIR}/${ALGORITHM}_attack.json"
