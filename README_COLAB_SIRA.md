# SIRA, CoDA, and SPIA on Google Colab L4

This adaptation runs the official SIRA workflow using the two smaller attack
models reported by the paper and compares them with the proposed Context-Anchor
Desynchronization Attack (CoDA) and Student-ID Prefix Injection Attack (SPIA).
It runs entirely on a Google Colab L4 GPU; the local laptop GPU is not used.

## Three Attack Methods

1. **SIRA** rewrites high-self-information tokens. It is evaluated with the
   paper's Tiny and Small model configurations.
2. **CoDA** changes low-information context anchors immediately before
   suspicious tokens.
3. **SPIA** prepends `student_id: 35571241` to the untouched watermarked text.

SIRA and CoDA responses do not contain the student-ID prefix.

The final dataframe has four rows because SIRA is evaluated with two model
configurations, but the `attack_method` column groups them into three methods.

[Open the paper-model notebook in Google Colab](https://colab.research.google.com/github/hanifnoerr/Self-information-Rewrite-Attack/blob/codex/browser-colab-l4/SIRA_COLAB_L4.ipynb)

## Two Models From the Paper

The paper reports three SIRA sizes. This L4 workflow runs the two that fit on a
single L4:

| Paper method | Released-code checkpoint | Precision | Paper KGW ASR |
|---|---|---|---:|
| SIRA-Tiny | `meta-llama/Llama-3.2-3B-Instruct` | bf16 | 96.4% |
| SIRA-Small | `meta-llama/Meta-Llama-3-8B-Instruct` | bf16 | 100.0% |

The exact default checkpoint names and loading settings are stored in:

```text
config/paper_models_l4.json
```

The paper calls the models Llama3 Instruct 3B and 8B. Meta did not release an
original Llama 3 3B checkpoint; the official SIRA `pre_attack.py` default
identifies Llama 3.2 3B Instruct as the intended SIRA-Tiny checkpoint. The
official README identifies Meta-Llama-3-8B-Instruct for SIRA-Small.

The previous Llama/Gemma/Qwen transfer configuration remains available as the
optional `config/model_matrix_l4.json`. Gemma 4 26B A4B has been removed.

## Hugging Face Access

Add a Colab Secret named `HF_TOKEN`.

Both Llama checkpoints require accepting their Hugging Face terms. A 403 does
not stop the notebook: the runner records that model as `skipped_access`,
continues, and keeps the skipped row in the final dataframe.

Accept access separately for
[`Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
and [`Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

## Run the Browser Notebook

1. Open the notebook using the link above.
2. Choose **Runtime > Change runtime type > L4 GPU**.
3. Add the approved `HF_TOKEN` Colab Secret.
4. Run all cells from top to bottom.
5. Start with `SAMPLES = 10`.
6. After the smoke test works, increase the sample count.
7. Copy results to Drive using the final notebook cell.
8. Use **Runtime > Disconnect and delete runtime**.

`RESET_OUTPUTS = True` is enabled by default so outputs from an older model run
cannot be silently reused.

## Optional Colab CLI Run

The official Colab CLI supports Linux and macOS. On Windows, run it from WSL or
use the browser notebook above.

From this repository, the non-interactive driver checks authentication, creates
the named `sira-l4` L4 session, uploads the adaptation, runs the two paper models,
downloads the results archive, and stops the session:

```bash
colab sessions
SAMPLES=10 ALGORITHM=KGW bash scripts/run_colab_l4_sira.sh
```

The script uses absolute `/content/...` paths and has an exit trap that calls:

```bash
colab stop -s sira-l4
```

## Paper Comparison Design

Every model receives:

- the same OPT-1.3B-generated watermarked text;
- the same C4 subset;
- the same watermark algorithm;
- the same SIRA threshold of 30;
- the same greedy rewrite workflow.

SPIA is an independent attack. It prepends the following line without rewriting
the original watermarked response:

```text
student_id: 35571241
```

Each model writes to `/content/sira_outputs/sira_models/<model-label>/`:

```text
/content/sira_outputs/sira_models/llama_3_2_3b/
/content/sira_outputs/sira_models/llama_3_8b/
```

## Metrics

The transfer report includes:

- attack success rate;
- average watermark score;
- semantic similarity to the original watermarked text;
- evaluated samples;
- failed samples;
- quantization and model family.
- CoDA average anchor count and anchor rate;
- SPIA attack-success and semantic-similarity results.
- attack-success difference between CoDA and SPIA.

The default strong-transfer criterion is:

```text
attack success rate >= 0.80
semantic similarity >= 0.75
```

This is an experiment criterion, not a standard established by the paper.

## CoDA

CoDA tests a different idea from SIRA:

1. Calculate token self-information.
2. Identify high-self-information target tokens using threshold 30.
3. Leave each target token visible.
4. Select a low-self-information token immediately before the target.
5. Replace only that preceding context anchor with a short natural alternative.

For the configured KGW watermark, `prefix_length` is 1. Its green-list decision
depends on the previous token, so changing the anchor can alter the green-list
context for the following suspicious token while preserving most visible text.

Example:

```text
Original: North America and Europe
CoDA:     North America as well as Europe
```

Run it directly:

```bash
python scripts/run_coda_attack.py \
  --input_path /content/sira_outputs/watermarked/KGW_response.json \
  --output_path /content/sira_outputs/coda/coda_attack.jsonl \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --threshold 30 \
  --dtype bf16 \
  --max_samples 10
```

Important limitation: an instruction-following model may make edits beyond the
marked anchors. Semantic similarity and the recorded anchor rate must therefore
be checked alongside attack success.

## SPIA

SPIA does not use an LLM. It leaves the watermarked response unchanged and
prepends one unrelated context line:

```text
student_id: 35571241
```

Run it directly:

```bash
python scripts/run_spia_attack.py \
  --input_path /content/sira_outputs/watermarked/KGW_response.json \
  --output_path /content/sira_outputs/spia/student_id_prefix_attack.jsonl \
  --max_samples 10
```

If SPIA lowers the detector score, the effect comes from context-prefix
injection rather than paraphrasing or anchor rewriting.

## Outputs

```text
/content/sira_outputs/model_runs.json
/content/sira_outputs/coda/coda_attack.jsonl
/content/sira_outputs/spia/student_id_prefix_attack.jsonl
/content/sira_outputs/results/transfer_eval.json
/content/sira_outputs/results/transfer_eval.csv
/content/sira_outputs/results/transfer_comparison.json
/content/sira_outputs/results/transfer_comparison.csv
/content/sira_outputs/results/coda_eval.json
/content/sira_outputs/results/coda_eval.csv
/content/sira_outputs/results/spia_eval.json
/content/sira_outputs/results/spia_eval.csv
/content/sira_outputs/results/paper_style_comparison.json
/content/sira_outputs/results/paper_style_comparison.csv
/content/sira_outputs/final_report.md
```

Paper-model logs:

```text
/content/sira_outputs/logs/sira_llama_3_2_3b.log
/content/sira_outputs/logs/sira_llama_3_8b.log
```

## Paper Comparison Caveats

The paper reports SIRA-Tiny using a 3B Llama3 Instruct model and SIRA-Small
using an 8B Llama3 Instruct model. It uses 500 random C4 samples, OPT-1.3B
watermark generation, bf16, and A100 GPUs.

The released `pre_attack.py` defaults to
`meta-llama/Llama-3.2-3B-Instruct`, which is the strongest evidence for the
intended SIRA-Tiny checkpoint.

A small L4 run is a functional paper-model comparison, not automatically an
exact reproduction. The paper uses 500 samples and A100 GPUs. Start with ten
samples to confirm the pipeline, then increase the sample count. The 8B
model runs in bf16 by default to match the paper more closely; if it exceeds
L4 memory, using 4-bit is a useful fallback but must be reported as a mismatch.
