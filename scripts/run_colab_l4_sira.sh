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
pip install --upgrade transformers accelerate
pip install bitsandbytes tqdm"

FILES_TO_UPLOAD=(
  "README_COLAB_SIRA.md"
  "requirements.txt"
  "config/paper_models_l4.json"
  "config/model_matrix_l4.json"
  "scripts/model_utils.py"
  "scripts/generate_watermark.py"
  "scripts/pre_attack.py"
  "scripts/attack.py"
  "scripts/prepare_colab_huggingface.py"
  "scripts/run_sira_model_l4.sh"
  "scripts/run_model_matrix_l4.py"
  "scripts/run_coda_attack.py"
  "scripts/evaluate_sira_transfer.py"
  "scripts/compare_transfer_results.py"
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
python scripts/run_model_matrix_l4.py \
  --config_path /content/Self-information-Rewrite-Attack/config/paper_models_l4.json \
  --repo_dir /content/Self-information-Rewrite-Attack \
  --output_root /content/sira_outputs \
  --algorithm '${ALGORITHM}' \
  --samples '${SAMPLES}'

python scripts/run_coda_attack.py \
  --input_path '/content/sira_outputs/watermarked/${ALGORITHM}_response.json' \
  --output_path /content/sira_outputs/coda/coda_attack.jsonl \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --threshold 30 \
  --dtype bf16 \
  --max_samples '${SAMPLES}'

python scripts/write_environment.py \
  --output_path /content/sira_outputs/environment.json \
  --models_config /content/sira_outputs/model_runs.json

python scripts/evaluate_sira_transfer.py \
  --generation_model facebook/opt-1.3b \
  --algorithm '${ALGORITHM}' \
  --watermarked_input '/content/sira_outputs/watermarked/${ALGORITHM}_response.json' \
  --models_config /content/sira_outputs/model_runs.json \
  --coda_input /content/sira_outputs/coda/coda_attack.jsonl \
  --output_root /content/sira_outputs \
  --dtype bf16 \
  --max_samples '${SAMPLES}'

python scripts/compare_transfer_results.py \
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
