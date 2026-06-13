import argparse
import csv
import json
import os


PAPER_RESULTS = {
    "SIRA-Tiny": {
        "KGW": 0.964,
        "Unigram": 0.876,
        "UPV": 0.844,
        "EWD": 0.978,
        "DIP": 0.998,
        "SIR": 0.750,
        "EXP": 0.906,
    },
    "SIRA-Small": {
        "KGW": 1.000,
        "Unigram": 0.938,
        "UPV": 0.930,
        "EWD": 1.000,
        "DIP": 0.998,
        "SIR": 0.834,
        "EXP": 0.934,
    },
}


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def save_csv(path, rows):
    fieldnames = list(rows[0].keys()) if rows else ["method"]
    with open(path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verdict(difference):
    if difference is None:
        return "not run"
    if abs(difference) < 0.0001:
        return "same"
    if abs(difference) <= 0.05:
        return "close"
    return "different"


def show(value):
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Compare reproduced SIRA and Base64 results with the paper.")
    parser.add_argument("--output_root", default="/content/sira_outputs")
    parser.add_argument("--algorithm", default="KGW")
    parser.add_argument("--dataset", default="C4 trim_c4_500.json")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--generation_model", default="facebook/opt-1.3b")
    parser.add_argument("--tiny_model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--small_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    results_dir = os.path.join(args.output_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    evaluation_rows = load_json(os.path.join(results_dir, "all_eval.json"), [])
    environment = load_json(os.path.join(args.output_root, "environment.json"), {})
    blocker_path = os.path.join(args.output_root, "logs", "colab_cli_blocker.txt")

    result_by_method = {row["method"]: row for row in evaluation_rows}
    comparison_rows = []

    for method in ["SIRA-Tiny", "SIRA-Small"]:
        paper_value = PAPER_RESULTS[method].get(args.algorithm)
        reproduced = result_by_method.get(method, {}).get("attack_success_rate")
        difference = reproduced - paper_value if reproduced is not None and paper_value is not None else None
        note = verdict(difference)
        expected_model = (
            "meta-llama/Llama-3.2-3B-Instruct"
            if method == "SIRA-Tiny"
            else "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        actual_model = args.tiny_model if method == "SIRA-Tiny" else args.small_model
        if actual_model != expected_model:
            note = (
                f"adapted experiment using {actual_model}, not the released-code checkpoint "
                f"{expected_model}; numerical equality would not be an exact reproduction"
            )
        if note != "same":
            note += (
                "; paper used 500 random C4 samples, OPT-1.3B watermark generation, "
                "bf16, and A100 GPUs. Check sample subset, checkpoint, quantization, and package versions."
            )
        comparison_rows.append(
            {
                "method": method,
                "watermark_algorithm": args.algorithm,
                "paper_attack_success_rate": paper_value,
                "reproduced_attack_success_rate": reproduced,
                "difference": difference,
                "note": note,
            }
        )

    save_json(os.path.join(results_dir, "paper_comparison.json"), comparison_rows)
    save_csv(os.path.join(results_dir, "paper_comparison.csv"), comparison_rows)

    final_table = []
    for row in evaluation_rows:
        final_table.append(
            {
                "method": row.get("method"),
                "detector_input": row.get("detector_input"),
                "attack_success_rate": row.get("attack_success_rate"),
                "semantic_similarity": row.get("semantic_similarity"),
                "decode_success_rate": row.get("decode_success_rate"),
                "note": row.get("note", ""),
            }
        )
    save_json(os.path.join(results_dir, "final_comparison.json"), final_table)
    save_csv(os.path.join(results_dir, "final_comparison.csv"), final_table)

    report_path = os.path.join(args.output_root, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("# SIRA Colab L4 Reproduction Report\n\n")
        report.write("## 1. Environment\n\n")
        report.write(f"- GPU: {environment.get('gpu', 'not recorded')}\n")
        report.write(f"- Python: {environment.get('python_version', 'not recorded')}\n")
        report.write(f"- PyTorch: {environment.get('torch_version', 'not recorded')}\n")
        report.write(f"- Transformers: {environment.get('transformers_version', 'not recorded')}\n")
        report.write(f"- Generation model: {args.generation_model}\n")
        report.write(f"- SIRA-Tiny: {args.tiny_model}, bf16 by default\n")
        report.write(f"- SIRA-Small: {args.small_model}, 4-bit NF4 by default on L4\n\n")

        report.write("## 2. Data\n\n")
        report.write(f"- Dataset: {args.dataset}\n")
        report.write(f"- Requested samples: {args.samples}\n")
        report.write(f"- Watermark algorithm: {args.algorithm}\n")
        report.write("- Paper setting: 500 random C4 test samples, 230 generated tokens, OPT-1.3B\n")
        report.write("- Official checked-in generator currently requests 200 new tokens, which is a known mismatch.\n\n")

        report.write("## 3. SIRA Results\n\n")
        if comparison_rows:
            report.write("| method | paper ASR | reproduced ASR | difference | assessment |\n")
            report.write("|---|---:|---:|---:|---|\n")
            for row in comparison_rows:
                report.write(
                    f"| {row['method']} | {show(row['paper_attack_success_rate'])} | "
                    f"{show(row['reproduced_attack_success_rate'])} | {show(row['difference'])} | "
                    f"{row['note']} |\n"
                )
        report.write("\n")

        report.write("## 4. Base64 Baseline\n\n")
        if final_table:
            report.write("| method | detector input | ASR | semantic similarity | decode success rate | note |\n")
            report.write("|---|---|---:|---:|---:|---|\n")
            for row in final_table:
                report.write(
                    f"| {row['method']} | {row['detector_input']} | {show(row['attack_success_rate'])} | "
                    f"{show(row['semantic_similarity'])} | {show(row['decode_success_rate'])} | "
                    f"{row['note']} |\n"
                )
        else:
            report.write("No Base64 evaluation results were available.\n")
        report.write("\n")

        report.write("## 5. Honest Conclusion\n\n")
        if not evaluation_rows:
            report.write(
                "The experiment was not reproduced because no completed evaluation files were available. "
                "Do not interpret missing values as successful attacks.\n\n"
            )
            if os.path.exists(blocker_path):
                with open(blocker_path, "r", encoding="utf-8") as blocker_file:
                    blocker_text = blocker_file.read().strip()
                report.write("### Execution Blocker\n\n```text\n")
                report.write(blocker_text)
                report.write("\n```\n\n")
        report.write(
            "Raw Base64 can bypass a detector that expects natural-language tokens, but it is only a formatting bypass. "
            "Decoded raw Base64 should reproduce the original watermark exactly. If an LLM-generated Base64 paraphrase "
            "works after decoding, the meaningful attack is the paraphrasing, not Base64 encoding itself.\n"
        )

    print(f"Saved paper comparison: {os.path.join(results_dir, 'paper_comparison.csv')}")
    print(f"Saved final comparison: {os.path.join(results_dir, 'final_comparison.csv')}")
    print(f"Final report: {report_path}")


if __name__ == "__main__":
    main()
