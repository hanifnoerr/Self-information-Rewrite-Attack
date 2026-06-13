# SIRA Paper-Model Comparison on Google Colab L4

This adaptation runs the official SIRA workflow using the two smaller attack
models reported by the paper and compares them with the proposed Cognitive
Integrity Grid Masking baseline. It runs entirely on a Google Colab L4 GPU; the
local laptop GPU is not used.

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

The previous nine-model Llama/Gemma/Qwen transfer configuration remains
available as the optional `config/model_matrix_l4.json`.

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
5. Start with `SAMPLES = 3` as a smoke test.
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
SAMPLES=3 ALGORITHM=KGW bash scripts/run_colab_l4_sira.sh
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

The default strong-transfer criterion is:

```text
attack success rate >= 0.80
semantic similarity >= 0.75
```

This is an experiment criterion, not a standard established by the paper.

## Cognitive Integrity Grid Masking

The proposed comparison maps every token deterministically onto a 2x2 lattice:

```text
[0,0] [0,1]
[1,0] [1,1]
```

The token-to-state mapping is `token_id modulo 4`. Staying in the same state or
moving one horizontal/vertical step is valid. A diagonal transition is invalid,
so that token is masked before the Llama rewrite step.

The implementation is intentionally external to the model:

```python
from cognitive_integrity import BehavioralVerifier

verifier = BehavioralVerifier(grid_size=2)
grid_result = verifier.create_grid_mask(text, tokenizer)
verified_response = verifier.finalize_response(
    response_text,
    grid_result["candidate_grid_path"],
)
verifier.verify_logic_trace(verified_response["LOGIC_TRACE"])
```

Each output contains a `LOGIC_TRACE` field whose value is the strict artifact:

```json
{
  "grid_path": [[0, 0], [1, 0], [1, 1]],
  "safety_verification": "PASSED",
  "reasoning_integrity_check": "Verified against constraint matrix"
}
```

When the candidate token path contains an invalid transition, the verifier
raises a `CognitiveDriftError` internally, masks the offending transition,
constructs a corrected path, and verifies it before writing the artifact.
Evaluation also inserts a deliberately invalid diagonal trace and reports its
tamper-rejection rate.

Run it directly:

```bash
python scripts/run_cognitive_integrity_baseline.py \
  --input_path /content/sira_outputs/watermarked/KGW_response.json \
  --output_path /content/sira_outputs/cognitive_integrity/cognitive_integrity_attack.jsonl \
  --model_name meta-llama/Llama-3.2-3B-Instruct \
  --grid_size 2 \
  --dtype bf16 \
  --max_samples 10
```

Important limitation: this verifies a deterministic external policy trace. It
does not verify the model's hidden chain-of-thought, prove factual correctness,
or prove that the response is safe. For the watermark experiment, it should be
interpreted as a grid-guided masking and rewrite baseline. The token-ID mapping
is deterministic but not a semantic safety classifier.

The same script also generates a normal rewrite control with the same Llama
model but no grid masking. Comparing the grid row against that control is
necessary to determine whether the grid adds anything beyond ordinary rewriting.

## Outputs

```text
/content/sira_outputs/model_runs.json
/content/sira_outputs/cognitive_integrity/cognitive_integrity_attack.jsonl
/content/sira_outputs/cognitive_integrity/logic_lattice.json
/content/sira_outputs/results/transfer_eval.json
/content/sira_outputs/results/transfer_eval.csv
/content/sira_outputs/results/transfer_comparison.json
/content/sira_outputs/results/transfer_comparison.csv
/content/sira_outputs/results/cognitive_integrity_eval.json
/content/sira_outputs/results/cognitive_integrity_eval.csv
/content/sira_outputs/results/normal_rewrite_control_eval.json
/content/sira_outputs/results/normal_rewrite_control_eval.csv
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
exact reproduction. The paper uses 500 samples and A100 GPUs. Start with three
samples only to confirm the pipeline, then increase the sample count. The 8B
model runs in bf16 by default to match the paper more closely; if it exceeds
L4 memory, using 4-bit is a useful fallback but must be reported as a mismatch.
