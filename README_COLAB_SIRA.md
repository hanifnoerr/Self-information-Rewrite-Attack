# SIRA Cross-Model Transfer Test on Google Colab L4

This adaptation tests whether the official SIRA attack workflow transfers across
different LLM families and compares it with the proposed Cognitive Integrity
Grid Masking baseline. It runs entirely on a Google Colab L4 GPU; the local
laptop GPU is not used.

[Open the transfer-test notebook in Google Colab](https://colab.research.google.com/github/hanifnoerr/Self-information-Rewrite-Attack/blob/codex/browser-colab-l4/SIRA_COLAB_L4.ipynb)

## Recommended Models

The notebook runs the same SIRA workflow on:

| Role | Model | Family | Precision |
|---|---|---|---|
| Released-code reference | `meta-llama/Llama-3.2-3B-Instruct` | Llama | bf16 |
| Non-Llama transfer test | `google/gemma-2-2b-it` | Gemma | bf16 |
| Non-Llama transfer test | `Qwen/Qwen2.5-7B-Instruct` | Qwen | 4-bit NF4 |

Why these models:

- Gemma tests transfer to a Google model family with a smaller text-only model
  that fits comfortably on one L4.
- Qwen tests a different model family and a larger attack model.
- The Llama checkpoint is the default in the released SIRA `pre_attack.py` and
  acts as the reference.

Testing Gemma and Qwen can provide evidence that SIRA transfers beyond Llama.
It cannot prove SIRA works on every possible LLM.

Qwen is intentionally larger, but it uses 4-bit loading to fit comfortably on
an L4. Treat its result as a practical transfer test, not a precision-controlled
comparison with the bf16 Llama and Gemma runs.

## Hugging Face Access

Add a Colab Secret named `HF_TOKEN`.

The Hugging Face account behind that token must be able to download:

```text
meta-llama/Llama-3.2-3B-Instruct
google/gemma-2-2b-it
```

Qwen is public. Gemma requires accepting Google's terms on Hugging Face. The
notebook tests an actual `config.json` download for every model before starting.

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
the named `sira-l4` L4 session, uploads the adaptation, runs the three models,
downloads the results archive, and stops the session:

```bash
colab sessions
SAMPLES=10 ALGORITHM=KGW bash scripts/run_colab_l4_sira.sh
```

The script uses absolute `/content/...` paths and has an exit trap that calls:

```bash
colab stop -s sira-l4
```

## Fair Transfer-Test Design

Every model receives:

- the same OPT-1.3B-generated watermarked text;
- the same C4 subset;
- the same watermark algorithm;
- the same SIRA threshold of 30;
- the same greedy rewrite workflow.

Each model writes to a separate directory:

```text
/content/sira_outputs/sira_models/llama_3_2_3b/
/content/sira_outputs/sira_models/gemma_2_2b/
/content/sira_outputs/sira_models/qwen_2_5_7b/
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

Per-model logs:

```text
/content/sira_outputs/logs/sira_llama_3_2_3b.log
/content/sira_outputs/logs/sira_gemma_2_2b.log
/content/sira_outputs/logs/sira_qwen_2_5_7b.log
```

## Paper Comparison Caveats

The paper reports SIRA-Tiny using a 3B Llama3 Instruct model and SIRA-Small
using an 8B Llama3 Instruct model. It uses 500 random C4 samples, OPT-1.3B
watermark generation, bf16, and A100 GPUs.

The released `pre_attack.py` defaults to
`meta-llama/Llama-3.2-3B-Instruct`, which is the strongest evidence for the
intended SIRA-Tiny checkpoint.

A 10-sample L4 run is a functional reproduction attempt, not an exact paper
reproduction. Gemma and Qwen results measure transferability and should not be
compared as exact SIRA-Tiny or SIRA-Small reproductions.
