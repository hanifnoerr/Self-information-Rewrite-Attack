# SIRA Reproduction on Google Colab CLI L4

This adaptation runs the official SIRA split workflow (`pre_attack.py` followed by
`attack.py`) and adds a Base64 baseline. It uses absolute `/content/...` paths and
non-interactive Colab CLI commands.

## Recommended for Windows: Browser Colab Notebook

Native Windows cannot run the official Colab CLI because it depends on Unix-only
modules. Your laptop's 2 GB GPU is not used when running in browser Colab.

Open the included notebook in Colab:

[Open SIRA_COLAB_L4.ipynb in Google Colab](https://colab.research.google.com/github/hanifnoerr/Self-information-Rewrite-Attack/blob/codex/browser-colab-l4/SIRA_COLAB_L4.ipynb)

In Colab, choose **Runtime > Change runtime type > L4 GPU**, add a Colab Secret
named `HF_TOKEN`, and run the cells from top to bottom. The notebook starts with
10 KGW samples and copies final outputs to Google Drive.

## Important Reproduction Notes

- The paper uses `facebook/opt-1.3b` to generate watermarked text, 500 random C4
  test samples, bf16, and A100 GPUs.
- The official README shows OPT-2.7B, which does not match the paper.
- The checked-in `generate_watermark.py` requests 200 new tokens even though the
  paper says it generates 230 tokens. The L4 runners use 230 by default; set
  `GENERATED_TOKENS=200` to test the untouched official default.
- The repository includes `dataset/c4/trim_c4_500.json`, but the paper only says
  "500 random samples"; it does not publish a seed that proves this is the exact
  paper subset.
- `attack_onestep.py` requires local filesystem model paths and rejects Hugging
  Face model names. This adaptation therefore uses the official split workflow.
- SIRA-Small uses 4-bit NF4 on an L4 by default. The paper used bf16, so a
  quantized SIRA-Small result must not be called identical without matching
  numbers.

Paper Table 1 attack-success-rate targets:

| method | KGW | Unigram | UPV | EWD | DIP | SIR | EXP |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIRA-Tiny | 96.4% | 87.6% | 84.4% | 97.8% | 99.8% | 75.0% | 90.6% |
| SIRA-Small | 100.0% | 93.8% | 93.0% | 100.0% | 99.8% | 83.4% | 93.4% |

## 1. Install and Authenticate Colab CLI

The official Colab CLI currently supports Linux and macOS, not native Windows.
On Linux/macOS:

```bash
uv tool install google-colab-cli
```

Configure Application Default Credentials with all required scopes:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Check authentication and current sessions:

```bash
colab sessions
colab whoami
```

## 2. Configure Hugging Face Access

Both requested Meta Llama checkpoints are gated. Accept their licenses on
Hugging Face, then add a Colab Secret named `HF_TOKEN` and grant notebook access.
The script `scripts/prepare_colab_huggingface.py` prints a clear error if the
token or model permission is missing.

Requested models:

```text
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Meta-Llama-3-8B-Instruct
```

## 3. Start an L4 Session

```bash
colab sessions
colab new -s sira-l4 --gpu L4
colab status -s sira-l4
```

Always use the named session `sira-l4`.

## 4. Clone and Install Inside Colab

Use a piped console command, never an interactive console:

```bash
printf '%s\n' 'set -e
cd /content
git clone https://github.com/Allencheng97/Self-information-Rewrite-Attack.git
cd /content/Self-information-Rewrite-Attack
pip install -r requirements.txt' | colab console -s sira-l4
```

The all-in-one local driver clones the repository and uploads this adaptation:

```bash
SAMPLES=10 ALGORITHM=KGW bash scripts/run_colab_l4_sira.sh
```

It checks auth, creates an L4, runs a smoke test, downloads an output archive,
exports logs, and stops the session even when a command fails.

## 5. Run SIRA-Tiny

Inside the Colab VM:

```bash
cd /content/Self-information-Rewrite-Attack
ALGORITHM=KGW SAMPLES=10 bash scripts/run_sira_tiny_l4.sh
```

Output:

```text
/content/sira_outputs/sira_tiny/
```

For a 500-sample run:

```bash
ALGORITHM=KGW SAMPLES=500 bash scripts/run_sira_tiny_l4.sh
```

## 6. Run SIRA-Small

SIRA-Small defaults to 4-bit NF4 so the 8B model fits comfortably on an L4:

```bash
cd /content/Self-information-Rewrite-Attack
ALGORITHM=KGW SAMPLES=10 bash scripts/run_sira_small_l4.sh
```

Output:

```text
/content/sira_outputs/sira_small/
```

Remove `--load_in_4bit` from the small runner only if the runtime has enough
memory and an exact bf16 comparison is required.

## 7. Run the Base64 Baseline

Raw Base64 plus LLM Base64 paraphrase:

```bash
export PYTHONPATH=/content/Self-information-Rewrite-Attack
python scripts/run_base64_baseline.py \
  --input_path /content/sira_outputs/watermarked/KGW_response.json \
  --output_root /content/sira_outputs \
  --algorithm KGW \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --dtype bf16 \
  --run_normal_paraphrase \
  --max_samples 10
```

Raw encoding outputs:

```text
/content/sira_outputs/base64/base64_raw.jsonl
/content/sira_outputs/base64/base64_decoded.jsonl
```

LLM Base64 output:

```text
/content/sira_outputs/base64_llm/base64_llm_raw.jsonl
```

## 8. Decode LLM Base64 Output

```bash
python scripts/decode_base64_outputs.py \
  --input_path /content/sira_outputs/base64_llm/base64_llm_raw.jsonl \
  --output_path /content/sira_outputs/base64_llm/base64_llm_decoded.jsonl
```

The decoder removes common Markdown/prefix mistakes, repairs padding, records
decode errors, and never silently treats a failed decode as success.

## 9. Evaluate SIRA and Base64

```bash
export PYTHONPATH=/content/Self-information-Rewrite-Attack
python scripts/evaluate_base64_baseline.py \
  --generation_model facebook/opt-1.3b \
  --algorithm KGW \
  --watermarked_input /content/sira_outputs/watermarked/KGW_response.json \
  --sira_tiny_input /content/sira_outputs/sira_tiny/final/KGW_attack.json \
  --sira_small_input /content/sira_outputs/sira_small/final/KGW_attack.json \
  --base64_raw_input /content/sira_outputs/base64/base64_raw.jsonl \
  --base64_decoded_input /content/sira_outputs/base64/base64_decoded.jsonl \
  --base64_llm_raw_input /content/sira_outputs/base64_llm/base64_llm_raw.jsonl \
  --base64_llm_decoded_input /content/sira_outputs/base64_llm/base64_llm_decoded.jsonl \
  --normal_paraphrase_input /content/sira_outputs/normal_paraphrase/normal_paraphrase.jsonl \
  --output_root /content/sira_outputs \
  --dtype bf16
```

Add `--calculate_ppl` when the extra runtime is acceptable.

Required result files are written to:

```text
/content/sira_outputs/results/sira_tiny_eval.json
/content/sira_outputs/results/sira_small_eval.json
/content/sira_outputs/results/sira_tiny_eval.csv
/content/sira_outputs/results/sira_small_eval.csv
/content/sira_outputs/base64_llm/base64_llm_eval.json
/content/sira_outputs/base64_llm/base64_llm_eval.csv
```

## 10. Compare with the Paper

```bash
python scripts/compare_results.py \
  --output_root /content/sira_outputs \
  --algorithm KGW \
  --samples 500
```

This writes:

```text
/content/sira_outputs/results/paper_comparison.json
/content/sira_outputs/results/paper_comparison.csv
/content/sira_outputs/results/final_comparison.json
/content/sira_outputs/results/final_comparison.csv
/content/sira_outputs/final_report.md
```

Interpretation:

- Raw Base64 fooling the detector is only a formatting bypass because it is not
  readable natural language.
- Decoding raw Base64 returns the original watermarked text, so its watermark
  should remain.
- If an LLM Base64 paraphrase works after decoding, paraphrasing is the actual
  attack mechanism.

## 11. Run Every Paper Watermark

Run each algorithm separately so failures and logs remain easy to inspect:

```bash
for algorithm in KGW Unigram UPV EWD DIP SIR EXP; do
  ALGORITHM="${algorithm}" SAMPLES=500 bash scripts/run_colab_l4_sira.sh
done
```

This can consume substantial Colab compute. Start with 5 or 10 samples.

## 12. List, Download, and Stop

List remote outputs:

```bash
printf '%s\n' 'find /content/sira_outputs -maxdepth 4 -type f -print | sort' \
  | colab console -s sira-l4
```

Archive and download:

```bash
printf '%s\n' 'tar -czf /content/sira_outputs.tar.gz -C /content sira_outputs' \
  | colab console -s sira-l4
colab download -s sira-l4 /content/sira_outputs.tar.gz ./sira_outputs.tar.gz
```

Export logs and always stop the session:

```bash
colab log -s sira-l4 -o colab_session.jsonl
colab log -s sira-l4 -o colab_session.md
colab stop -s sira-l4
```
