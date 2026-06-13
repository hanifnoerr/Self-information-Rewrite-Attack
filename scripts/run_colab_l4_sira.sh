#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="sira-l4"
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_OUTPUT="${LOCAL_REPO}/sira_outputs"
LOCAL_LOG_DIR="${LOCAL_OUTPUT}/logs"
SAMPLES="${SAMPLES:-10}"
ALGORITHM="${ALGORITHM:-KGW}"

mkdir -p "${LOCAL_LOG_DIR}"
RUN_LOG="${LOCAL_LOG_DIR}/colab_run.log"

stop_session() {
  echo "Stopping Colab session ${SESSION_NAME}" | tee -a "${RUN_LOG}"
  colab stop -s "${SESSION_NAME}" 2>&1 | tee -a "${RUN_LOG}" || true
}

run_remote() {
  local command="$1"
  printf '%s\n' "${command}" | colab console -s "${SESSION_NAME}" 2>&1 | tee -a "${RUN_LOG}"
}

trap stop_session EXIT

echo "Checking Colab authentication and sessions" | tee "${RUN_LOG}"
colab sessions 2>&1 | tee -a "${RUN_LOG}"
colab whoami 2>&1 | tee -a "${RUN_LOG}"

echo "Creating named L4 session: ${SESSION_NAME}" | tee -a "${RUN_LOG}"
colab new -s "${SESSION_NAME}" --gpu L4 2>&1 | tee -a "${RUN_LOG}"
colab status -s "${SESSION_NAME}" 2>&1 | tee -a "${RUN_LOG}"

run_remote "set -euo pipefail
cd /content
if [ ! -d /content/Self-information-Rewrite-Attack/.git ]; then
  git clone https://github.com/Allencheng97/Self-information-Rewrite-Attack.git
fi
cd /content/Self-information-Rewrite-Attack
pip install -r requirements.txt
pip install bitsandbytes tqdm"

FILES_TO_UPLOAD=(
  "README_COLAB_SIRA.md"
  "requirements.txt"
  "scripts/model_utils.py"
  "scripts/generate_watermark.py"
  "scripts/pre_attack.py"
  "scripts/attack.py"
  "scripts/prepare_colab_huggingface.py"
  "scripts/run_sira_tiny_l4.sh"
  "scripts/run_sira_small_l4.sh"
  "scripts/run_base64_baseline.py"
  "scripts/decode_base64_outputs.py"
  "scripts/create_base64_side_by_side.py"
  "scripts/evaluate_base64_baseline.py"
  "scripts/compare_results.py"
  "scripts/write_environment.py"
)

echo "Uploading adapted files" | tee -a "${RUN_LOG}"
for relative_path in "${FILES_TO_UPLOAD[@]}"; do
  colab upload -s "${SESSION_NAME}" \
    "${LOCAL_REPO}/${relative_path}" \
    "/content/Self-information-Rewrite-Attack/${relative_path}" 2>&1 | tee -a "${RUN_LOG}"
done

colab exec -s "${SESSION_NAME}" -f "${LOCAL_REPO}/scripts/prepare_colab_huggingface.py" 2>&1 | tee -a "${RUN_LOG}"

run_remote "set -euo pipefail
cd /content/Self-information-Rewrite-Attack
export PYTHONPATH=/content/Self-information-Rewrite-Attack
python scripts/write_environment.py
ALGORITHM='${ALGORITHM}' SAMPLES='${SAMPLES}' bash scripts/run_sira_tiny_l4.sh
ALGORITHM='${ALGORITHM}' SAMPLES='${SAMPLES}' bash scripts/run_sira_small_l4.sh

python scripts/run_base64_baseline.py \
  --input_path '/content/sira_outputs/watermarked/${ALGORITHM}_response.json' \
  --output_root /content/sira_outputs \
  --algorithm '${ALGORITHM}' \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --dtype bf16 \
  --run_normal_paraphrase \
  --max_samples '${SAMPLES}'

python scripts/decode_base64_outputs.py \
  --input_path /content/sira_outputs/base64_llm/base64_llm_raw.jsonl \
  --output_path /content/sira_outputs/base64_llm/base64_llm_decoded.jsonl

python scripts/create_base64_side_by_side.py \
  --raw_original /content/sira_outputs/base64/base64_raw.jsonl \
  --decoded_original /content/sira_outputs/base64/base64_decoded.jsonl \
  --llm_raw /content/sira_outputs/base64_llm/base64_llm_raw.jsonl \
  --llm_decoded /content/sira_outputs/base64_llm/base64_llm_decoded.jsonl \
  --normal_paraphrase /content/sira_outputs/normal_paraphrase/normal_paraphrase.jsonl \
  --external_raw /content/sira_outputs/base64_llm/normal_paraphrase_base64_raw.jsonl \
  --external_decoded /content/sira_outputs/base64_llm/normal_paraphrase_base64_decoded.jsonl \
  --output_jsonl /content/sira_outputs/base64_llm/base64_side_by_side.jsonl \
  --output_csv /content/sira_outputs/base64_llm/base64_side_by_side.csv

python scripts/evaluate_base64_baseline.py \
  --generation_model facebook/opt-1.3b \
  --algorithm '${ALGORITHM}' \
  --watermarked_input '/content/sira_outputs/watermarked/${ALGORITHM}_response.json' \
  --sira_tiny_input '/content/sira_outputs/sira_tiny/final/${ALGORITHM}_attack.json' \
  --sira_small_input '/content/sira_outputs/sira_small/final/${ALGORITHM}_attack.json' \
  --base64_raw_input /content/sira_outputs/base64/base64_raw.jsonl \
  --base64_decoded_input /content/sira_outputs/base64/base64_decoded.jsonl \
  --base64_llm_raw_input /content/sira_outputs/base64_llm/base64_llm_raw.jsonl \
  --base64_llm_decoded_input /content/sira_outputs/base64_llm/base64_llm_decoded.jsonl \
  --normal_paraphrase_input /content/sira_outputs/normal_paraphrase/normal_paraphrase.jsonl \
  --normal_paraphrase_base64_raw_input /content/sira_outputs/base64_llm/normal_paraphrase_base64_raw.jsonl \
  --normal_paraphrase_base64_decoded_input /content/sira_outputs/base64_llm/normal_paraphrase_base64_decoded.jsonl \
  --output_root /content/sira_outputs \
  --dtype bf16 \
  --max_samples '${SAMPLES}'

python scripts/compare_results.py \
  --output_root /content/sira_outputs \
  --algorithm '${ALGORITHM}' \
  --samples '${SAMPLES}'

find /content/sira_outputs -maxdepth 4 -type f -printf '%p\n' | sort
tar -czf /content/sira_outputs.tar.gz -C /content sira_outputs"

colab log -s "${SESSION_NAME}" -o "${LOCAL_LOG_DIR}/colab_session.jsonl"
colab log -s "${SESSION_NAME}" -o "${LOCAL_LOG_DIR}/colab_session.md"
colab download -s "${SESSION_NAME}" /content/sira_outputs.tar.gz "${LOCAL_OUTPUT}/${ALGORITHM}_sira_outputs.tar.gz"

echo "Downloaded results archive: ${LOCAL_OUTPUT}/${ALGORITHM}_sira_outputs.tar.gz" | tee -a "${RUN_LOG}"
echo "Remote final report: /content/sira_outputs/final_report.md" | tee -a "${RUN_LOG}"
