#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/content/Self-information-Rewrite-Attack"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/sira_outputs}"
ALGORITHM="${ALGORITHM:-KGW}"
SAMPLES="${SAMPLES:-500}"
GENERATION_MODEL="${GENERATION_MODEL:-facebook/opt-1.3b}"
ATTACK_MODEL="${SMALL_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
GENERATED_TOKENS="${GENERATED_TOKENS:-230}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"

WATERMARK_DIR="${OUTPUT_ROOT}/watermarked"
SMALL_DIR="${OUTPUT_ROOT}/sira_small"
INTERMEDIATE_DIR="${SMALL_DIR}/intermediate"
FINAL_DIR="${SMALL_DIR}/final"

mkdir -p "${WATERMARK_DIR}" "${INTERMEDIATE_DIR}" "${FINAL_DIR}" "${OUTPUT_ROOT}/logs"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

LOG_FILE="${OUTPUT_ROOT}/logs/sira_small.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; echo "[ERROR] SIRA-Small failed at line ${LINENO} with exit status ${status}. Log: ${LOG_FILE}"; exit ${status}' ERR

echo "Running SIRA-Small with ${ATTACK_MODEL}"
QUANTIZATION_ARGS=()
if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
  QUANTIZATION_ARGS+=(--load_in_4bit)
fi

echo "Algorithm=${ALGORITHM}, samples=${SAMPLES}, threshold=30, load_in_4bit=${LOAD_IN_4BIT}"

echo "=== Stage 0: Generate watermarked data ==="
python scripts/generate_watermark.py \
  --algorithms "${ALGORITHM}" \
  --model_path "${GENERATION_MODEL}" \
  --input_path "/content/Self-information-Rewrite-Attack/dataset/c4/trim_c4_500.json" \
  --output_dir "${WATERMARK_DIR}" \
  --cuda_visible_devices 0 \
  --cuda_device 0 \
  --dtype bf16 \
  --max_samples "${SAMPLES}" \
  --max_new_tokens "${GENERATED_TOKENS}" \
  --min_length "${GENERATED_TOKENS}" \
  --seed 42

echo "=== Stage 1 and 2: Paraphrase and self-information blanking ==="
python scripts/pre_attack.py \
  --input_dir "${WATERMARK_DIR}" \
  --result_dir "${INTERMEDIATE_DIR}" \
  --model_path "${ATTACK_MODEL}" \
  --threshold 30 \
  --gpu 0 \
  --algorithms "${ALGORITHM}" \
  --dtype bf16 \
  "${QUANTIZATION_ARGS[@]}" \
  --max_samples "${SAMPLES}" \
  --seed 42

echo "=== Stage 3: Rewrite attack ==="
python scripts/attack.py \
  --model_path "${ATTACK_MODEL}" \
  --input_dir "${INTERMEDIATE_DIR}" \
  --output_dir "${FINAL_DIR}" \
  --gpu 0 \
  --algorithms "${ALGORITHM}" \
  --dtype bf16 \
  "${QUANTIZATION_ARGS[@]}" \
  --max_samples "${SAMPLES}" \
  --seed 42

echo "SIRA-Small output: ${FINAL_DIR}/${ALGORITHM}_attack.json"
