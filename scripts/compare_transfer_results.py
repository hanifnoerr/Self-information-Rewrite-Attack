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

PAPER_SMALL_ASR = {
    "KGW": 1.000,
    "Unigram": 0.938,
    "UPV": 0.930,
    "EWD": 1.000,
    "DIP": 0.998,
    "SIR": 0.834,
    "EXP": 0.934,
}


def get_paper_reference(label, algorithm):
    if label == "llama_3_2_3b":
        return "SIRA-Tiny", PAPER_TINY_ASR.get(algorithm)
    if label == "llama_3_8b":
        return "SIRA-Small", PAPER_SMALL_ASR.get(algorithm)
    return None, None


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


def subtract_if_measured(first_value, second_value):
    if first_value is None or second_value is None:
        return None
    return round(first_value - second_value, 6)


def average_measured(values):
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return sum(measured) / len(measured)


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
    parser.add_argument("--samples", type=int, default=500)
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

    sira_rows = [
        row for row in comparison_rows
        if row.get("method_type") == "SIRA"
    ]
    sira_labels = {row.get("label") for row in sira_rows}
    paper_reproduction_mode = bool(sira_labels) and sira_labels.issubset(
        {"llama_3_2_3b", "llama_3_8b"}
    )
    coda_rows = [
        row for row in comparison_rows
        if row.get("method_type") == "CoDA"
    ]
    passing_non_llama = [
        row for row in comparison_rows
        if row.get("method_type") == "SIRA"
        and row.get("model_family") != "Llama"
        and row.get("attack_success_rate") is not None
        and row["transfer_threshold_pass"]
    ]
    tested_non_llama = [
        row for row in comparison_rows
        if row.get("method_type") == "SIRA"
        and row.get("model_family") != "Llama"
        and row.get("attack_success_rate") is not None
    ]
    sira_by_label = {row.get("label"): row for row in sira_rows}
    coda_by_label = {
        row.get("source_model_label"): row for row in coda_rows
    }

    paper_style_rows = []
    for row in comparison_rows:
        paper_method, paper_asr = None, None
        if row.get("method_type") == "SIRA":
            paper_method, paper_asr = get_paper_reference(row.get("label"), args.algorithm)
        note = "Cross-model SIRA transfer test."
        if paper_method:
            note = (
                f"Released-code checkpoint for paper {paper_method}; compare cautiously unless "
                "all paper settings match."
            )
        elif row.get("method_type") == "CoDA":
            note = (
                "Proposed CoDA method. It changes low-information anchor tokens before "
                "high-information targets; it was not evaluated by the paper."
            )
        if row.get("run_status") in {"skipped_access", "failed"}:
            error_lines = row.get("run_error", "").splitlines()
            error_summary = error_lines[0] if error_lines else "No error details recorded."
            note = f"{row.get('run_status')}: {error_summary}"

        reproduced_asr = row.get("attack_success_rate")
        difference = subtract_if_measured(reproduced_asr, paper_asr)
        source_model_label = row.get("source_model_label", row.get("label"))
        paired_sira_row = sira_by_label.get(source_model_label)
        paired_coda_row = coda_by_label.get(source_model_label)
        paired_sira_asr = (
            paired_sira_row.get("attack_success_rate") if paired_sira_row else None
        )
        paired_coda_asr = (
            paired_coda_row.get("attack_success_rate") if paired_coda_row else None
        )

        paper_style_rows.append(
            {
                "method": row.get("display_name"),
                "attack_method": row.get("method_type"),
                "paper_method": paper_method,
                "model_family": row.get("model_family"),
                "parameter_size": row.get("parameter_size"),
                "size_tier": row.get("size_tier"),
                "quantization": row.get("quantization"),
                "run_status": row.get("run_status", "completed"),
                "watermark_algorithm": args.algorithm,
                "paper_attack_success_rate": paper_asr,
                "reproduced_attack_success_rate": reproduced_asr,
                "difference": difference,
                "paired_sira_asr": paired_sira_asr,
                "paired_coda_asr": paired_coda_asr,
                "coda_minus_sira_asr": subtract_if_measured(paired_coda_asr, paired_sira_asr),
                "semantic_similarity": row.get("semantic_similarity"),
                "average_anchor_count": row.get("average_anchor_count"),
                "average_anchor_rate": row.get("average_anchor_rate"),
                "average_suspicious_token_count": row.get("average_suspicious_token_count"),
                "note": note,
            }
        )

    save_csv(os.path.join(results_dir, "paper_style_comparison.csv"), paper_style_rows)
    with open(os.path.join(results_dir, "paper_style_comparison.json"), "w", encoding="utf-8") as output_file:
        json.dump(paper_style_rows, output_file, indent=2)

    report_path = os.path.join(args.output_root, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as report:
        if paper_reproduction_mode:
            report.write("# SIRA Paper-Model Reproduction Report\n\n")
        else:
            report.write("# SIRA Cross-Model Transfer Report\n\n")
        report.write("## Goal\n\n")
        if paper_reproduction_mode:
            report.write(
                "Compare the released-code SIRA-Tiny and SIRA-Small Llama checkpoints against "
                "the attack success rates reported by the paper, then compare the proposed CoDA "
                "anchor-desynchronization attack against them.\n\n"
            )
        else:
            report.write(
                "Test whether the same SIRA attack pipeline transfers across different LLM "
                "families and practical size tiers. This can provide evidence of transferability, "
                "but it cannot prove SIRA works on every LLM.\n\n"
            )

        report.write("## Experiment\n\n")
        report.write(f"- Watermark algorithm: {args.algorithm}\n")
        report.write(f"- Samples: {args.samples}\n")
        report.write("- Shared watermarked data: OPT-1.3B generation on the same C4 subset\n")
        report.write("- SIRA threshold: 30\n")
        report.write("- Compared methods: SIRA and CoDA\n")
        if paper_reproduction_mode:
            report.write("- Attack models: paper SIRA-Tiny 3B and SIRA-Small 8B configurations\n")
        else:
            report.write("- Attack models: configured cross-model transfer matrix\n")
        report.write(
            f"- Strong-transfer criterion: ASR >= {args.asr_threshold:.2f} and "
            f"semantic similarity >= {args.similarity_threshold:.2f}\n\n"
        )
        report.write(
            "Model size and quantization are recorded because they are possible confounders. "
            "In particular, a 4-bit result should not be treated as a precision-controlled "
            "comparison with a bf16 result.\n\n"
        )
        if paper_reproduction_mode:
            report.write(
                "The paper describes Llama3 Instruct models with 3B and 8B parameters. The "
                "released code identifies the intended 3B checkpoint as Llama 3.2 3B Instruct "
                "and the 8B checkpoint as Meta-Llama-3-8B-Instruct.\n\n"
            )
        else:
            report.write(
                "The configured tiers are practical L4 tiers rather than perfectly "
                "parameter-matched controls.\n\n"
            )

        report.write("## Results\n\n")
        report.write("| attack method | configuration | family | tier | size | quantization | status | ASR | semantic similarity | average watermark score | failures | comparison criterion |\n")
        report.write("|---|---|---|---|---:|---|---|---:|---:|---:|---:|---|\n")
        for row in comparison_rows:
            report.write(
                f"| {row.get('method_type', 'N/A')} | {row['display_name']} | {row['model_family']} | "
                f"{row.get('size_tier', 'N/A')} | {row['parameter_size']} | {row['quantization']} | "
                f"{row.get('run_status', 'completed')} | "
                f"{show(row['attack_success_rate'])} | "
                f"{show(row['semantic_similarity'])} | {show(row['average_watermark_score'])} | "
                f"{row['failed_samples']} | {'pass' if row['transfer_threshold_pass'] else 'fail'} |\n"
            )
        report.write("\n")

        report.write("## Paper Reference\n\n")
        for paper_label in ("llama_3_2_3b", "llama_3_8b"):
            paper_method, paper_asr = get_paper_reference(paper_label, args.algorithm)
            reproduced_row = next(
                (row for row in sira_rows if row.get("label") == paper_label),
                None,
            )
            if not reproduced_row or paper_asr is None:
                continue
            reproduced_asr = reproduced_row.get("attack_success_rate")
            report.write(f"- Paper {paper_method} {args.algorithm} ASR: {paper_asr:.4f}\n")
            report.write(f"- Reproduced {paper_method} ASR: {show(reproduced_asr)}\n")
            report.write(
                f"- Difference: {show(subtract_if_measured(reproduced_asr, paper_asr))}\n"
            )
        if sira_rows:
            report.write(
                "\n- These are not exact reproductions unless sample count, subset, hardware, "
                "precision, package versions, and all generation settings match the paper.\n\n"
            )

        report.write("## CoDA Across Models\n\n")
        if coda_rows:
            report.write("| model | family | CoDA ASR | SIRA ASR | CoDA minus SIRA | similarity | anchor rate |\n")
            report.write("|---|---|---:|---:|---:|---:|---:|\n")
            for row in coda_rows:
                paired_sira_row = sira_by_label.get(row.get("source_model_label"))
                paired_sira_asr = (
                    paired_sira_row.get("attack_success_rate") if paired_sira_row else None
                )
                report.write(
                    f"| {row['display_name']} | {row['model_family']} | "
                    f"{show(row.get('attack_success_rate'))} | {show(paired_sira_asr)} | "
                    f"{show(subtract_if_measured(row.get('attack_success_rate'), paired_sira_asr))} | "
                    f"{show(row.get('semantic_similarity'))} | {show(row.get('average_anchor_rate'))} |\n"
                )
            report.write("\n")
            report.write(
                f"- Average CoDA ASR: {show(average_measured([row.get('attack_success_rate') for row in coda_rows]))}\n"
            )
            report.write(
                f"- Average CoDA semantic similarity: "
                f"{show(average_measured([row.get('semantic_similarity') for row in coda_rows]))}\n\n"
            )
            report.write(
                "CoDA leaves each selected high-self-information target token visible and asks "
                "the rewrite model to replace a low-self-information anchor immediately before "
                "it. For KGW with prefix length 1, changing the previous token changes the "
                "green-list context for the following token. CoDA is a proposed method, so no "
                "paper attack-success value exists for it.\n\n"
            )
        else:
            report.write("CoDA was not evaluated.\n\n")

        report.write("## Honest Conclusion\n\n")
        measured_coda_rows = [
            row for row in coda_rows
            if row.get("attack_success_rate") is not None
        ]
        if measured_coda_rows:
            best_coda_row = max(
                measured_coda_rows,
                key=lambda row: row.get("attack_success_rate"),
            )
            paired_wins = 0
            paired_tests = 0
            for row in measured_coda_rows:
                paired_sira_row = sira_by_label.get(row.get("source_model_label"))
                if paired_sira_row and paired_sira_row.get("attack_success_rate") is not None:
                    paired_tests += 1
                    if row["attack_success_rate"] > paired_sira_row["attack_success_rate"]:
                        paired_wins += 1
            report.write(
                f"Best CoDA configuration: {best_coda_row['display_name']} with ASR "
                f"{show(best_coda_row.get('attack_success_rate'))} and semantic similarity "
                f"{show(best_coda_row.get('semantic_similarity'))}. "
            )
            if best_coda_row.get("transfer_threshold_pass"):
                report.write(
                    "It met the experiment's attack-success and semantic-similarity criterion. "
                    "This is preliminary evidence for the anchor-desynchronization idea, not a "
                    "confirmed result at the current sample count.\n\n"
                )
            else:
                report.write(
                    "It did not meet the experiment's combined attack-success and "
                    "semantic-similarity criterion in this run.\n\n"
                )
            report.write(
                f"CoDA beat SIRA on {paired_wins} of {paired_tests} measured same-model pairs.\n\n"
            )
        if paper_reproduction_mode:
            report.write(
                "This run directly tests the two smaller attack-model configurations reported "
                "by the paper. Interpret similarity to the paper only after checking both the "
                "sample count and the numerical ASR differences above.\n"
            )
        elif tested_non_llama and len(passing_non_llama) == len(tested_non_llama):
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
    print(f"Saved paper-style comparison: {os.path.join(results_dir, 'paper_style_comparison.csv')}")
    print(f"Final report: {report_path}")


if __name__ == "__main__":
    main()
