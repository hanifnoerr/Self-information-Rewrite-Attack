import argparse
import csv
import json
import os


PAPER_TINY_ASR = {
    "KGW": 0.964,
    "Unigram": 0.876,
    "UPV": 0.844,
    "EWD": 0.978,
    "DIP": 0.998,
    "SIR": 0.750,
    "EXP": 0.906,
}


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def show(value):
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["label"]

    with open(path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create a cross-model SIRA transfer report.")
    parser.add_argument("--output_root", default="/content/sira_outputs")
    parser.add_argument("--algorithm", default="KGW")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--asr_threshold", type=float, default=0.8)
    parser.add_argument("--similarity_threshold", type=float, default=0.75)
    args = parser.parse_args()

    results_path = os.path.join(args.output_root, "results", "transfer_eval.json")
    results = load_json(results_path, [])
    attack_results = [row for row in results if row.get("label") != "watermarked_original"]

    comparison_rows = []
    for row in attack_results:
        asr = row.get("attack_success_rate")
        similarity = row.get("semantic_similarity")
        passes = (
            asr is not None
            and similarity is not None
            and asr >= args.asr_threshold
            and similarity >= args.similarity_threshold
        )
        comparison_rows.append(
            {
                **row,
                "transfer_threshold_pass": passes,
                "threshold_note": (
                    f"Requires ASR >= {args.asr_threshold:.2f} and similarity >= {args.similarity_threshold:.2f}"
                ),
            }
        )

    results_dir = os.path.join(args.output_root, "results")
    save_csv(os.path.join(results_dir, "transfer_comparison.csv"), comparison_rows)
    with open(os.path.join(results_dir, "transfer_comparison.json"), "w", encoding="utf-8") as output_file:
        json.dump(comparison_rows, output_file, indent=2)

    passing_non_llama = [
        row for row in comparison_rows
        if row.get("model_family") != "Llama" and row["transfer_threshold_pass"]
    ]
    tested_non_llama = [
        row for row in comparison_rows
        if row.get("model_family") != "Llama"
    ]

    report_path = os.path.join(args.output_root, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write("# SIRA Cross-Model Transfer Report\n\n")
        report.write("## Goal\n\n")
        report.write(
            "Test whether the SIRA attack pipeline transfers from the released Llama checkpoint "
            "to attack models from other LLM families. This experiment can provide evidence of "
            "transferability, but testing a few models cannot prove SIRA works on any LLM.\n\n"
        )

        report.write("## Experiment\n\n")
        report.write(f"- Watermark algorithm: {args.algorithm}\n")
        report.write(f"- Samples: {args.samples}\n")
        report.write("- Shared watermarked data: OPT-1.3B generation on the same C4 subset\n")
        report.write("- SIRA threshold: 30\n")
        report.write(
            f"- Strong-transfer criterion: ASR >= {args.asr_threshold:.2f} and "
            f"semantic similarity >= {args.similarity_threshold:.2f}\n\n"
        )
        report.write(
            "Model size and quantization are recorded because they are possible confounders. "
            "In particular, a 4-bit result should not be treated as a precision-controlled "
            "comparison with a bf16 result.\n\n"
        )

        report.write("## Results\n\n")
        report.write("| attack model | family | size | quantization | ASR | semantic similarity | average watermark score | failures | transfer criterion |\n")
        report.write("|---|---|---:|---|---:|---:|---:|---:|---|\n")
        for row in comparison_rows:
            report.write(
                f"| {row['display_name']} | {row['model_family']} | {row['parameter_size']} | "
                f"{row['quantization']} | {show(row['attack_success_rate'])} | "
                f"{show(row['semantic_similarity'])} | {show(row['average_watermark_score'])} | "
                f"{row['failed_samples']} | {'pass' if row['transfer_threshold_pass'] else 'fail'} |\n"
            )
        report.write("\n")

        llama_result = next((row for row in comparison_rows if row.get("model_family") == "Llama"), None)
        paper_asr = PAPER_TINY_ASR.get(args.algorithm)
        report.write("## Paper Reference\n\n")
        if llama_result and llama_result.get("attack_success_rate") is not None and paper_asr is not None:
            difference = llama_result["attack_success_rate"] - paper_asr
            report.write(f"- Paper SIRA-Tiny {args.algorithm} ASR: {paper_asr:.4f}\n")
            report.write(f"- This Llama run ASR: {llama_result['attack_success_rate']:.4f}\n")
            report.write(f"- Difference: {difference:.4f}\n")
            report.write(
                "- This is not an exact reproduction unless sample count, subset, hardware, "
                "precision, package versions, and all generation settings match the paper.\n\n"
            )

        report.write("## Honest Conclusion\n\n")
        if tested_non_llama and len(passing_non_llama) == len(tested_non_llama):
            report.write(
                "Both tested non-Llama model families met the chosen attack-success and semantic-similarity "
                "thresholds. This is strong evidence that SIRA transfers beyond Llama for these settings. "
                "It does not prove compatibility with every LLM.\n"
            )
        elif passing_non_llama:
            report.write(
                "SIRA transferred strongly to some, but not all, tested non-Llama model families. "
                "The claim of universal model compatibility is not supported by this run.\n"
            )
        else:
            report.write(
                "The tested non-Llama model families did not meet the chosen transfer criterion. "
                "This run does not support a broad model-agnostic SIRA claim.\n"
            )

    print(f"Saved transfer comparison: {os.path.join(results_dir, 'transfer_comparison.csv')}")
    print(f"Final report: {report_path}")


if __name__ == "__main__":
    main()
