import argparse
import difflib
import json
import re
from pathlib import Path


MODEL_LAYOUT = [
    {
        "label": "llama_3_8b",
        "name": "Llama 3 8B Instruct",
    },
    {
        "label": "qwen_2_5_7b",
        "name": "Qwen 2.5 7B Instruct",
    },
    {
        "label": "mistral_7b_v0_3",
        "name": "Mistral 7B Instruct v0.3",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an HTML dashboard for saved SIRA results."
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--visualization_dir",
        default="",
        help="Defaults to <output_root>/visualization",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        default=[0, 100, 200, 300, 400],
    )
    parser.add_argument(
        "--kgw_tokens_path",
        default="",
        help=(
            "Exact KGW token export. Defaults to "
            "<output_root>/visualization/kgw_token_colors.json when present."
        ),
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def load_experiment_data(output_root):
    watermarked_rows = read_jsonl(
        output_root / "watermarked" / "KGW_response.json"
    )
    model_data = {}

    for model in MODEL_LAYOUT:
        model_root = output_root / "sira_models" / model["label"]
        model_data[model["label"]] = {
            "name": model["name"],
            "blank_rows": read_jsonl(
                model_root / "intermediate" / "KGW_blank.json"
            ),
            "attack_rows": read_jsonl(
                model_root / "final" / "KGW_attack.json"
            ),
        }

    evaluation_rows = read_json(output_root / "results" / "transfer_eval.json")
    return watermarked_rows, model_data, evaluation_rows


def build_summary(evaluation_rows):
    summary = []
    for row in evaluation_rows:
        if row.get("label") == "watermarked_original":
            summary.append(
                {
                    "label": row["label"],
                    "method": "Input",
                    "name": row["display_name"],
                    "attack_success_rate": row["attack_success_rate"],
                    "semantic_similarity": row["semantic_similarity"],
                    "average_watermark_score": row["average_watermark_score"],
                }
            )
        elif row.get("method_type") in {"SIRA", "CoDA"}:
            summary.append(
                {
                    "label": row["label"],
                    "source_model_label": row.get(
                        "source_model_label",
                        row.get("label"),
                    ),
                    "method": row["method_type"],
                    "name": row["display_name"],
                    "attack_success_rate": row["attack_success_rate"],
                    "semantic_similarity": row["semantic_similarity"],
                    "average_watermark_score": row[
                        "average_watermark_score"
                    ],
                }
            )
    return summary


def validate_sample_ids(sample_ids, row_count):
    for sample_id in sample_ids:
        if sample_id < 0 or sample_id >= row_count:
            raise ValueError(
                f"Sample ID {sample_id} is outside the available range "
                f"0 to {row_count - 1}."
            )


def split_words(text):
    return re.findall(r"\s+|[^\s]+", text)


def build_word_diff(original_text, attack_text):
    original_parts = split_words(original_text)
    attack_parts = split_words(attack_text)
    matcher = difflib.SequenceMatcher(
        None,
        original_parts,
        attack_parts,
        autojunk=False,
    )

    original_chunks = []
    attack_chunks = []
    retained_nonspace = 0
    original_nonspace = sum(
        1 for part in original_parts if not part.isspace()
    )
    attack_nonspace = sum(
        1 for part in attack_parts if not part.isspace()
    )

    for tag, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if tag == "equal":
            original_class = "same"
            attack_class = "same"
            retained_nonspace += sum(
                1
                for part in original_parts[first_start:first_end]
                if not part.isspace()
            )
        elif tag == "delete":
            original_class = "changed-original"
            attack_class = "same"
        elif tag == "insert":
            original_class = "same"
            attack_class = "changed-attack"
        else:
            original_class = "changed-original"
            attack_class = "changed-attack"

        if first_start != first_end:
            original_chunks.append(
                {
                    "text": "".join(
                        original_parts[first_start:first_end]
                    ),
                    "class": original_class,
                }
            )
        if second_start != second_end:
            attack_chunks.append(
                {
                    "text": "".join(
                        attack_parts[second_start:second_end]
                    ),
                    "class": attack_class,
                }
            )

    denominator = max(original_nonspace, attack_nonspace, 1)
    lexical_retention = retained_nonspace / denominator
    return original_chunks, attack_chunks, lexical_retention


def build_examples(
    watermarked_rows,
    model_data,
    sample_ids,
    kgw_token_data=None,
):
    examples = []
    for sample_id in sample_ids:
        source = watermarked_rows[sample_id]
        watermarked_text = source["watermarked_text"]
        kgw_example = None
        if kgw_token_data:
            kgw_example = kgw_token_data.get("examples", {}).get(
                str(sample_id)
            )
        example = {
            "id": sample_id,
            "prompt": source.get("prompt", ""),
            "watermarked_text": watermarked_text,
            "unwatermarked_text": source.get("unwatermarked_text", ""),
            "kgw_watermarked": (
                kgw_example.get("watermarked")
                if kgw_example
                else None
            ),
            "models": {},
        }

        for model in MODEL_LAYOUT:
            label = model["label"]
            blank_row = model_data[label]["blank_rows"][sample_id]
            attack_row = model_data[label]["attack_rows"][sample_id]
            if attack_row["watermarked_text"] != watermarked_text:
                raise ValueError(
                    f"Input mismatch for {label}, sample {sample_id}."
                )

            original_diff, attack_diff, lexical_retention = build_word_diff(
                watermarked_text,
                attack_row["attack_text"],
            )
            example["models"][label] = {
                "name": model["name"],
                "blank_text": blank_row["blank_text"],
                "reference_text": blank_row.get("ref_text", ""),
                "attack_text": attack_row["attack_text"],
                "original_diff": original_diff,
                "attack_diff": attack_diff,
                "kgw_attack": (
                    kgw_example.get("models", {}).get(label)
                    if kgw_example
                    else None
                ),
                "lexical_retention": lexical_retention,
                "mask_marker_count": blank_row["blank_text"].count("_"),
                "mask_group_count": len(
                    re.findall(r"_+", blank_row["blank_text"])
                ),
            }

        examples.append(example)
    return examples


def create_metrics_plot(summary, models, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed; skipping metrics_overview.png.")
        return

    sira_rows = [row for row in summary if row["method"] == "SIRA"]
    coda_rows = [row for row in summary if row["method"] == "CoDA"]
    input_row = next(row for row in summary if row["method"] == "Input")
    colors = ["#3168a8", "#408b8d", "#b78530"]
    short_names = [model["short_name"] for model in models]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.2),
        gridspec_kw={"width_ratios": [1.05, 0.9, 1.05]},
    )
    figure.patch.set_facecolor("#f4f6f9")

    scatter_axis = axes[0]
    for index, row in enumerate(sira_rows):
        scatter_axis.scatter(
            row["semantic_similarity"],
            row["attack_success_rate"],
            s=120,
            color=colors[index],
            edgecolor="white",
            linewidth=1.8,
            zorder=3,
        )
        scatter_axis.annotate(
            short_names[index],
            (
                row["semantic_similarity"],
                row["attack_success_rate"],
            ),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=10,
            weight="semibold",
        )
    scatter_axis.set_xlim(0.79, 0.88)
    scatter_axis.set_ylim(0.94, 1.0)
    scatter_axis.set_xlabel("Semantic similarity")
    scatter_axis.set_ylabel("Attack success rate")
    scatter_axis.set_title(
        "Attack success vs semantic similarity",
        loc="left",
        weight="bold",
    )
    scatter_axis.grid(color="#dce2ea", linewidth=0.8)

    score_axis = axes[1]
    score_rows = [input_row] + sira_rows
    score_names = ["Watermarked input"] + short_names
    score_values = [
        row["average_watermark_score"]
        for row in score_rows
    ]
    score_colors = ["#758195"] + colors
    bars = score_axis.barh(
        score_names,
        score_values,
        color=score_colors,
        height=0.55,
    )
    score_axis.invert_yaxis()
    score_axis.set_xlim(0, 9)
    score_axis.set_xlabel("Average KGW score")
    score_axis.set_title(
        "Residual watermark evidence",
        loc="left",
        weight="bold",
    )
    score_axis.grid(axis="x", color="#dce2ea", linewidth=0.8)
    for bar, value in zip(bars, score_values):
        score_axis.text(
            value + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )

    method_axis = axes[2]
    positions = list(range(len(models)))
    sira_values = [row["attack_success_rate"] for row in sira_rows]
    coda_by_label = {
        row["source_model_label"]: row
        for row in coda_rows
    }
    coda_values = [
        coda_by_label[model["label"]]["attack_success_rate"]
        for model in models
    ]
    width = 0.36
    method_axis.bar(
        [position - width / 2 for position in positions],
        sira_values,
        width,
        label="SIRA",
        color="#3168a8",
    )
    method_axis.bar(
        [position + width / 2 for position in positions],
        coda_values,
        width,
        label="Context-anchor baseline",
        color="#a4adba",
    )
    method_axis.set_xticks(positions, short_names)
    method_axis.set_ylim(0, 1.05)
    method_axis.set_ylabel("Attack success rate")
    method_axis.set_title(
        "Method comparison",
        loc="left",
        weight="bold",
    )
    method_axis.grid(axis="y", color="#dce2ea", linewidth=0.8)
    method_axis.legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.set_facecolor("white")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.suptitle(
        "SIRA cross-model evaluation on 500 KGW-watermarked samples",
        x=0.03,
        y=1.01,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.03,
        -0.015,
        "Semantic similarity is an embedding-based similarity score, not a factual-accuracy measurement.",
        fontsize=9,
        color="#5f6978",
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def build_html(data):
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SIRA Result Visualization</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #657085;
      --line: #dfe4ec;
      --surface: #ffffff;
      --surface-soft: #f6f8fb;
      --navy: #17365d;
      --blue: #3168a8;
      --teal: #408b8d;
      --gold: #b78530;
      --green: #dff2e3;
      --green-ink: #176338;
      --red: #f7dfdf;
      --red-ink: #8c2e2e;
      --change: #dce8f7;
      --change-ink: #244f7d;
      --kgw-green: #dff2e3;
      --kgw-green-ink: #176338;
      --kgw-red: #f7dfdf;
      --kgw-red-ink: #8c2e2e;
      --kgw-neutral: #e6e9ef;
      --mask: #e6e9ef;
      --mask-ink: #414c5f;
      --shadow: 0 12px 34px rgba(24, 39, 75, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #eef2f7;
      font-family: Inter, "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    header {{
      padding: 54px 7vw 48px;
      color: white;
      background:
        radial-gradient(circle at 85% 20%, rgba(255,255,255,.13), transparent 28%),
        linear-gradient(130deg, #102947, #1d4977 58%, #2c6797);
    }}
    header h1 {{
      margin: 0 0 10px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2rem, 4vw, 3.5rem);
      font-weight: 600;
      letter-spacing: -0.025em;
    }}
    header p {{
      max-width: 840px;
      margin: 0;
      color: #dce9f5;
      font-size: 1.06rem;
    }}
    main {{
      width: min(1500px, 94vw);
      margin: -22px auto 60px;
    }}
    .card {{
      margin-bottom: 22px;
      padding: 24px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
    }}
    h2 {{
      margin: 0 0 16px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.7rem;
      font-weight: 600;
    }}
    h3 {{
      margin: 0;
      font-size: 1rem;
      letter-spacing: .01em;
    }}
    .method-note {{
      padding: 14px 16px;
      color: #334056;
      background: #edf4fb;
      border-left: 4px solid var(--blue);
      border-radius: 5px;
      font-size: .94rem;
    }}
    .color-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin: 14px 0 2px;
      color: var(--muted);
      font-size: .86rem;
    }}
    .color-key {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .color-box {{
      width: 16px;
      height: 16px;
      border: 1px solid rgba(0,0,0,.08);
      border-radius: 4px;
    }}
    .color-box.green {{ background: var(--kgw-green); }}
    .color-box.red {{ background: var(--kgw-red); }}
    .color-box.neutral {{ background: var(--kgw-neutral); }}
    .color-box.prompt {{
      background: white;
      border-bottom: 3px solid var(--blue);
    }}
    .charts {{
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 18px;
    }}
    .chart-card {{
      min-width: 0;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: var(--surface-soft);
    }}
    .chart-card h3 {{ margin-bottom: 5px; }}
    .chart-card p {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: .85rem;
    }}
    svg {{
      display: block;
      width: 100%;
      min-height: 310px;
      overflow: visible;
    }}
    .axis {{ stroke: #9ca8b7; stroke-width: 1; }}
    .grid {{ stroke: #dce2ea; stroke-width: 1; }}
    .axis-label {{
      fill: #5c6879;
      font-size: 12px;
      font-family: Inter, "Segoe UI", sans-serif;
    }}
    .point-label {{
      fill: #253247;
      font-size: 12px;
      font-weight: 650;
      font-family: Inter, "Segoe UI", sans-serif;
    }}
    .bar-label {{
      fill: #344158;
      font-size: 12px;
      font-family: Inter, "Segoe UI", sans-serif;
    }}
    .chart-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      color: var(--muted);
      font-size: .82rem;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .swatch {{
      width: 13px;
      height: 13px;
      border-radius: 3px;
    }}
    .swatch.asr {{ background: var(--blue); }}
    .swatch.similarity {{ background: var(--teal); }}
    .swatch.score {{ background: var(--gold); }}
    .summary-table {{
      width: 100%;
      margin-top: 20px;
      border-collapse: collapse;
      font-size: .9rem;
    }}
    .summary-table th,
    .summary-table td {{
      padding: 10px 11px;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }}
    .summary-table th {{
      color: var(--muted);
      font-size: .75rem;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
      align-items: end;
    }}
    label {{
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: .78rem;
      font-weight: 700;
      letter-spacing: .06em;
      text-transform: uppercase;
    }}
    select {{
      width: 100%;
      padding: 11px 12px;
      color: var(--ink);
      background: white;
      border: 1px solid #bcc6d3;
      border-radius: 7px;
      font: inherit;
    }}
    .prompt {{
      margin-top: 16px;
      padding: 13px 15px;
      background: #f7f9fc;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .process {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin: 18px 0;
    }}
    .process-step {{
      padding: 13px 14px;
      background: var(--surface-soft);
      border: 1px solid var(--line);
      border-radius: 9px;
    }}
    .process-step span {{
      display: block;
      color: var(--blue);
      font-size: .72rem;
      font-weight: 700;
      letter-spacing: .07em;
      text-transform: uppercase;
    }}
    .process-step strong {{ font-size: .92rem; }}
    .comparison {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    .text-panel {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 11px;
      overflow: hidden;
      background: white;
    }}
    .panel-head {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      padding: 13px 15px;
      background: var(--surface-soft);
      border-bottom: 1px solid var(--line);
    }}
    .badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .badge {{
      padding: 3px 8px;
      border-radius: 99px;
      background: #e7edf5;
      color: #48566c;
      font-size: .73rem;
      font-weight: 650;
    }}
    .text-body {{
      min-height: 330px;
      max-height: 560px;
      overflow: auto;
      padding: 17px;
      white-space: pre-wrap;
      font-family: Georgia, "Times New Roman", serif;
      font-size: .94rem;
      line-height: 1.72;
    }}
    .diff.same {{ color: inherit; }}
    .diff.changed-original {{
      color: var(--red-ink);
      background: var(--red);
      border-radius: 3px;
    }}
    .diff.changed-attack {{
      color: var(--change-ink);
      background: var(--change);
      border-radius: 3px;
    }}
    .token {{
      padding: 1px 0;
      border-radius: 3px;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
    }}
    .token.green {{
      color: var(--kgw-green-ink);
      background: var(--kgw-green);
    }}
    .token.red {{
      color: var(--kgw-red-ink);
      background: var(--kgw-red);
    }}
    .token.neutral {{
      color: #515d6e;
      background: var(--kgw-neutral);
    }}
    .token.prompt {{
      border-bottom: 2px solid var(--blue);
    }}
    .mask-panel {{
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 11px;
      overflow: hidden;
    }}
    .mask-text {{
      max-height: 320px;
      overflow: auto;
      padding: 17px;
      white-space: pre-wrap;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: .88rem;
      line-height: 1.75;
    }}
    .mask-run {{
      padding: 1px 2px;
      color: var(--mask-ink);
      background: var(--mask);
      border-radius: 3px;
    }}
    details {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fafbfd;
    }}
    summary {{
      padding: 11px 14px;
      cursor: pointer;
      color: #35445a;
      font-weight: 650;
    }}
    details .detail-body {{
      padding: 0 14px 14px;
      white-space: pre-wrap;
      color: #445168;
    }}
    footer {{
      padding: 25px;
      text-align: center;
      color: var(--muted);
      font-size: .84rem;
    }}
    @media (max-width: 960px) {{
      .charts, .comparison, .controls, .process {{
        grid-template-columns: 1fr;
      }}
      .text-body {{ min-height: 240px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SIRA Result Visualization</h1>
    <p>
      Verified aggregate metrics and representative text examples from the
      500-sample KGW experiment.
    </p>
  </header>

  <main>
    <section class="card">
      <h2>Cross-model performance</h2>
      <p class="method-note">
        Attack success rate measures how often the rewritten text falls below
        the KGW detection threshold. Semantic similarity measures meaning
        preservation using sentence embeddings. A strong result is high on both
        axes. <span id="color-status"></span>
      </p>
      <div id="color-legend" class="color-legend"></div>
      <div class="charts">
        <article class="chart-card">
          <h3>Attack success versus semantic similarity</h3>
          <p>Upper-right is preferred. Values are aggregate results over 500 samples.</p>
          <svg id="scatter-chart" viewBox="0 0 620 360" role="img"></svg>
        </article>
        <article class="chart-card">
          <h3>Average KGW watermark score</h3>
          <p>Lower values indicate less detectable watermark evidence.</p>
          <svg id="score-chart" viewBox="0 0 540 360" role="img"></svg>
        </article>
      </div>
      <div class="chart-legend">
        <span class="legend-item"><span class="swatch asr"></span>Attack success rate</span>
        <span class="legend-item"><span class="swatch similarity"></span>Semantic similarity</span>
        <span class="legend-item"><span class="swatch score"></span>Average watermark score</span>
      </div>
      <table class="summary-table" id="summary-table"></table>
    </section>

    <section class="card">
      <h2>SIRA and context-anchor baseline</h2>
      <p>
        This chart compares attack success using the same three rewrite models.
        The context-anchor baseline preserved meaning well but removed less KGW
        evidence than SIRA.
      </p>
      <div class="chart-card">
        <svg id="method-chart" viewBox="0 0 900 360" role="img"></svg>
      </div>
    </section>

    <section class="card">
      <h2>Example comparison</h2>
      <div class="controls">
        <div>
          <label for="sample-select">Sample</label>
          <select id="sample-select"></select>
        </div>
        <div>
          <label for="model-select">Attack model</label>
          <select id="model-select"></select>
        </div>
        <div>
          <label for="display-select">Text highlighting</label>
          <select id="display-select"></select>
        </div>
      </div>
      <div id="example"></div>
    </section>
  </main>

  <footer>
    Generated directly from the saved JSON and CSV experiment outputs.
  </footer>

  <script id="result-data" type="application/json">{payload}</script>
  <script>
    const data = JSON.parse(document.getElementById("result-data").textContent);
    const sampleSelect = document.getElementById("sample-select");
    const modelSelect = document.getElementById("model-select");
    const displaySelect = document.getElementById("display-select");
    const exampleRoot = document.getElementById("example");
    const colors = ["#3168a8", "#408b8d", "#b78530"];

    const number = value => value == null ? "N/A" : Number(value).toFixed(3);
    const percent = value => value == null ? "N/A" : `${{(100 * value).toFixed(1)}}%`;
    const esc = value => String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
    const siraRows = data.summary.filter(row => row.method === "SIRA");
    const codaRows = data.summary.filter(row => row.method === "CoDA");
    const inputRow = data.summary.find(row => row.method === "Input");

    function renderColorStatus() {{
      const status = document.getElementById("color-status");
      const legend = document.getElementById("color-legend");
      if (data.kgw_available) {{
        status.textContent =
          "Exact KGW token classes were exported on the recorded A100 path and are available in the example viewer.";
        legend.innerHTML = `
          <span class="color-key"><span class="color-box green"></span>Green-list token</span>
          <span class="color-key"><span class="color-box red"></span>Red-list token</span>
          <span class="color-key"><span class="color-box neutral"></span>Prefix token, not scored</span>
          <span class="color-key"><span class="color-box prompt"></span>Original prompt span</span>`;
      }} else {{
        status.textContent =
          "Exact KGW token classes are not included. The example viewer uses lexical-change highlighting until an A100 token export is supplied.";
        legend.innerHTML = `
          <span class="color-key"><span class="color-box red"></span>Changed or removed wording</span>
          <span class="color-key"><span class="color-box" style="background:#dce8f7"></span>New wording</span>`;
      }}
    }}

    function svgText(x, y, text, className, anchor = "start") {{
      return `<text x="${{x}}" y="${{y}}" text-anchor="${{anchor}}" class="${{className}}">${{esc(text)}}</text>`;
    }}

    function renderScatter() {{
      const svg = document.getElementById("scatter-chart");
      const width = 620, height = 360;
      const left = 68, right = 28, top = 28, bottom = 58;
      const xMin = 0.79, xMax = 0.88, yMin = 0.94, yMax = 1.0;
      const x = value => left + (value - xMin) / (xMax - xMin) * (width - left - right);
      const y = value => height - bottom - (value - yMin) / (yMax - yMin) * (height - top - bottom);
      let out = "";

      [0.80, 0.82, 0.84, 0.86, 0.88].forEach(value => {{
        const px = x(value);
        out += `<line x1="${{px}}" y1="${{top}}" x2="${{px}}" y2="${{height-bottom}}" class="grid"/>`;
        out += svgText(px, height - 34, value.toFixed(2), "axis-label", "middle");
      }});
      [0.94, 0.96, 0.98, 1.00].forEach(value => {{
        const py = y(value);
        out += `<line x1="${{left}}" y1="${{py}}" x2="${{width-right}}" y2="${{py}}" class="grid"/>`;
        out += svgText(left - 10, py + 4, value.toFixed(2), "axis-label", "end");
      }});
      out += `<line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{height-bottom}}" class="axis"/>`;
      out += `<line x1="${{left}}" y1="${{height-bottom}}" x2="${{width-right}}" y2="${{height-bottom}}" class="axis"/>`;
      out += svgText((left + width - right) / 2, height - 8, "Semantic similarity", "axis-label", "middle");
      out += `<text x="17" y="${{height/2}}" transform="rotate(-90 17 ${{height/2}})" text-anchor="middle" class="axis-label">Attack success rate</text>`;

      siraRows.forEach((row, index) => {{
        const px = x(row.semantic_similarity);
        const py = y(row.attack_success_rate);
        const shortName = data.models.find(model => model.label === row.label)?.short_name || row.name;
        out += `<circle cx="${{px}}" cy="${{py}}" r="9" fill="${{colors[index]}}" stroke="white" stroke-width="3"/>`;
        out += svgText(px + 13, py - 10, shortName, "point-label");
        out += svgText(px + 13, py + 7, `${{percent(row.attack_success_rate)}} / ${{percent(row.semantic_similarity)}}`, "axis-label");
      }});
      svg.innerHTML = out;
    }}

    function renderScoreChart() {{
      const svg = document.getElementById("score-chart");
      const rows = [inputRow, ...siraRows];
      const width = 540, height = 360;
      const left = 150, right = 35, top = 28, bottom = 38;
      const max = 9;
      const x = value => left + value / max * (width - left - right);
      let out = "";

      [0, 2, 4, 6, 8].forEach(value => {{
        const px = x(value);
        out += `<line x1="${{px}}" y1="${{top}}" x2="${{px}}" y2="${{height-bottom}}" class="grid"/>`;
        out += svgText(px, height - 14, value, "axis-label", "middle");
      }});
      rows.forEach((row, index) => {{
        const y = 55 + index * 70;
        const label = index === 0 ? "Watermarked input" : data.models[index - 1].short_name;
        const value = row.average_watermark_score;
        const fill = index === 0 ? "#758195" : colors[index - 1];
        out += svgText(left - 12, y + 18, label, "bar-label", "end");
        out += `<rect x="${{left}}" y="${{y}}" width="${{x(value)-left}}" height="28" rx="5" fill="${{fill}}"/>`;
        out += svgText(x(value) + 8, y + 19, number(value), "point-label");
      }});
      svg.innerHTML = out;
    }}

    function renderMethodChart() {{
      const svg = document.getElementById("method-chart");
      const width = 900, height = 360;
      const left = 150, right = 40, top = 28, bottom = 45;
      const x = value => left + value * (width - left - right);
      let out = "";

      [0, .2, .4, .6, .8, 1].forEach(value => {{
        const px = x(value);
        out += `<line x1="${{px}}" y1="${{top}}" x2="${{px}}" y2="${{height-bottom}}" class="grid"/>`;
        out += svgText(px, height - 18, percent(value), "axis-label", "middle");
      }});
      data.models.forEach((model, index) => {{
        const sira = siraRows.find(row => row.label === model.label);
        const coda = codaRows.find(row => row.source_model_label === model.label);
        const y = 55 + index * 87;
        out += svgText(left - 12, y + 28, model.short_name, "bar-label", "end");
        out += `<rect x="${{left}}" y="${{y}}" width="${{x(sira.attack_success_rate)-left}}" height="23" rx="4" fill="#3168a8"/>`;
        out += `<rect x="${{left}}" y="${{y+31}}" width="${{x(coda.attack_success_rate)-left}}" height="23" rx="4" fill="#9aa7b8"/>`;
        out += svgText(x(sira.attack_success_rate) + 7, y + 17, `SIRA ${{percent(sira.attack_success_rate)}}`, "point-label");
        out += svgText(x(coda.attack_success_rate) + 7, y + 48, `Context-anchor ${{percent(coda.attack_success_rate)}}`, "axis-label");
      }});
      svg.innerHTML = out;
    }}

    function renderSummaryTable() {{
      document.getElementById("summary-table").innerHTML = `
        <thead><tr>
          <th>Model</th>
          <th>Attack success</th>
          <th>Semantic similarity</th>
          <th>Average KGW score</th>
        </tr></thead>
        <tbody>
          ${{siraRows.map(row => `
            <tr>
              <td>${{esc(row.name)}}</td>
              <td>${{percent(row.attack_success_rate)}}</td>
              <td>${{percent(row.semantic_similarity)}}</td>
              <td>${{number(row.average_watermark_score)}}</td>
            </tr>`).join("")}}
        </tbody>`;
    }}

    function renderChunks(chunks) {{
      return chunks.map(chunk =>
        `<span class="diff ${{chunk.class}}">${{esc(chunk.text)}}</span>`
      ).join("");
    }}

    function renderKgwTokens(result) {{
      return result.tokens.map(token => {{
        const promptClass = token.prompt ? " prompt" : "";
        return `<span class="token ${{token.class}}${{promptClass}}">${{esc(token.text)}}</span>`;
      }}).join("");
    }}

    function renderKgwBadges(result) {{
      const status = result.is_watermarked
        ? "Watermark detected"
        : "Below threshold";
      return `
        <span class="badge">${{status}}</span>
        <span class="badge">z = ${{number(result.score)}}</span>
        <span class="badge">green = ${{percent(result.green_ratio)}}</span>`;
    }}

    function renderMask(text) {{
      return String(text).split(/(_+)/).map(piece =>
        /^_+$/.test(piece)
          ? `<span class="mask-run">${{esc(piece)}}</span>`
          : esc(piece)
      ).join("");
    }}

    function renderExample() {{
      const example = data.examples[Number(sampleSelect.value)];
      const model = example.models[modelSelect.value];
      const metrics = siraRows.find(row => row.label === modelSelect.value);
      const showKgw =
        displaySelect.value === "kgw" &&
        example.kgw_watermarked &&
        model.kgw_attack;
      const originalBody = showKgw
        ? renderKgwTokens(example.kgw_watermarked)
        : renderChunks(model.original_diff);
      const attackBody = showKgw
        ? renderKgwTokens(model.kgw_attack)
        : renderChunks(model.attack_diff);
      const originalBadges = showKgw
        ? renderKgwBadges(example.kgw_watermarked)
        : `<span class="badge">Red highlight: changed or removed wording</span>`;
      const attackBadges = showKgw
        ? renderKgwBadges(model.kgw_attack)
        : `<span class="badge">Blue highlight: new wording</span>`;

      exampleRoot.innerHTML = `
        <div class="prompt"><strong>Prompt:</strong> ${{esc(example.prompt)}}</div>
        <div class="process">
          <div class="process-step"><span>Stage 1</span><strong>KGW-watermarked response</strong></div>
          <div class="process-step"><span>Stage 2</span><strong>High-self-information tokens masked</strong></div>
          <div class="process-step"><span>Stage 3</span><strong>${{esc(model.name)}} reconstruction</strong></div>
        </div>
        <div class="comparison">
          <article class="text-panel">
            <div class="panel-head">
              <h3>Watermarked response</h3>
              <div class="badges">${{originalBadges}}</div>
            </div>
            <div class="text-body">${{originalBody}}</div>
          </article>
          <article class="text-panel">
            <div class="panel-head">
              <h3>SIRA attack output</h3>
              <div class="badges">
                ${{attackBadges}}
                <span class="badge">Aggregate ASR ${{percent(metrics.attack_success_rate)}}</span>
                <span class="badge">Similarity ${{percent(metrics.semantic_similarity)}}</span>
              </div>
            </div>
            <div class="text-body">${{attackBody}}</div>
          </article>
        </div>
        <div class="mask-panel">
          <div class="panel-head">
            <h3>Masked intermediate text</h3>
            <div class="badges">
              <span class="badge">${{model.mask_marker_count}} underscore markers</span>
              <span class="badge">${{model.mask_group_count}} masked groups</span>
              <span class="badge">Lexical retention ${{percent(model.lexical_retention)}}</span>
            </div>
          </div>
          <div class="mask-text">${{renderMask(model.blank_text)}}</div>
        </div>
        <details>
          <summary>Reference paraphrase used during reconstruction</summary>
          <div class="detail-body">${{esc(model.reference_text)}}</div>
        </details>
        <details>
          <summary>Unwatermarked generation control</summary>
          <div class="detail-body">${{esc(example.unwatermarked_text)}}</div>
        </details>`;
    }}

    data.examples.forEach((example, index) => {{
      const option = document.createElement("option");
      option.value = index;
      option.textContent = `Sample ${{example.id}}`;
      sampleSelect.appendChild(option);
    }});
    data.models.forEach(model => {{
      const option = document.createElement("option");
      option.value = model.label;
      option.textContent = model.name;
      modelSelect.appendChild(option);
    }});
    if (data.kgw_available) {{
      const kgwOption = document.createElement("option");
      kgwOption.value = "kgw";
      kgwOption.textContent = "KGW red/green tokens";
      displaySelect.appendChild(kgwOption);
    }}
    const lexicalOption = document.createElement("option");
    lexicalOption.value = "lexical";
    lexicalOption.textContent = "Lexical changes";
    displaySelect.appendChild(lexicalOption);

    sampleSelect.addEventListener("change", renderExample);
    modelSelect.addEventListener("change", renderExample);
    displaySelect.addEventListener("change", renderExample);
    renderColorStatus();
    renderScatter();
    renderScoreChart();
    renderMethodChart();
    renderSummaryTable();
    renderExample();
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    visualization_dir = (
        Path(args.visualization_dir).resolve()
        if args.visualization_dir
        else output_root / "visualization"
    )
    visualization_dir.mkdir(parents=True, exist_ok=True)
    kgw_tokens_path = (
        Path(args.kgw_tokens_path).resolve()
        if args.kgw_tokens_path
        else visualization_dir / "kgw_token_colors.json"
    )
    kgw_token_data = (
        read_json(kgw_tokens_path)
        if kgw_tokens_path.exists()
        else None
    )

    watermarked_rows, model_data, evaluation_rows = load_experiment_data(
        output_root
    )
    validate_sample_ids(args.sample_ids, len(watermarked_rows))
    examples = build_examples(
        watermarked_rows,
        model_data,
        args.sample_ids,
        kgw_token_data=kgw_token_data,
    )

    models = []
    short_names = {
        "llama_3_8b": "Llama 8B",
        "qwen_2_5_7b": "Qwen 7B",
        "mistral_7b_v0_3": "Mistral 7B",
    }
    for model in MODEL_LAYOUT:
        models.append(
            {
                **model,
                "short_name": short_names[model["label"]],
            }
        )

    data = {
        "title": "SIRA Result Visualization",
        "models": models,
        "summary": build_summary(evaluation_rows),
        "examples": examples,
        "kgw_available": kgw_token_data is not None,
        "kgw_metadata": (
            kgw_token_data.get("metadata")
            if kgw_token_data
            else None
        ),
    }

    data_path = visualization_dir / "visualization_data.json"
    html_path = visualization_dir / "index.html"
    plot_path = visualization_dir / "metrics_overview.png"
    with open(data_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)
    html_path.write_text(build_html(data), encoding="utf-8")
    create_metrics_plot(data["summary"], data["models"], plot_path)

    print(f"Visualization data: {data_path}")
    print(f"Dashboard: {html_path}")
    if kgw_token_data:
        print(f"KGW token colors: {kgw_tokens_path}")
    else:
        print(
            "KGW token colors: not available; "
            "using lexical-change highlighting."
        )
    if plot_path.exists():
        print(f"Metrics plot: {plot_path}")


if __name__ == "__main__":
    main()
