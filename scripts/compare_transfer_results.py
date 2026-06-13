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
    return first_value - second_value


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

    sira_rows = [
        row for row in comparison_rows
        if row.get("method_type") == "SIRA"
    ]
    sira_labels = {row.get("label") for row in sira_rows}
    paper_reproduction_mode = bool(sira_labels) and sira_labels.issubset(
        {"llama_3_2_3b", "llama_3_8b"}
    )
    cognitive_row = next(
        (row for row in comparison_rows if row.get("label") == "cognitive_integrity_grid"),
        None,
    )
    control_row = next(
        (row for row in comparison_rows if row.get("label") == "normal_rewrite_control"),
        None,
    )
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

    paper_style_rows = []
    for row in comparison_rows:
        paper_method, paper_asr = get_paper_reference(row.get("label"), args.algorithm)
        note = "Cross-model SIRA transfer test."
        if paper_method:
            note = (
                f"Released-code checkpoint for paper {paper_method}; compare cautiously unless "
                "all paper settings match."
            )
        elif row.get("label") == "cognitive_integrity_grid":
            note = (
                "Grid-guided masking baseline. Trace verification measures structural "
                "consistency, not hidden reasoning correctness or safety."
            )
        elif row.get("label") == "normal_rewrite_control":
            note = "Same Llama rewrite model without grid masking."
        if row.get("run_status") in {"skipped_access", "failed"}:
            error_lines = row.get("run_error", "").splitlines()
            error_summary = error_lines[0] if error_lines else "No error details recorded."
            note = f"{row.get('run_status')}: {error_summary}"

        reproduced_asr = row.get("attack_success_rate")
        difference = None
        if paper_asr is not None and reproduced_asr is not None:
            difference = reproduced_asr - paper_asr

        paper_style_rows.append(
            {
                "method": row.get("display_name"),
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
                "semantic_similarity": row.get("semantic_similarity"),
                "logic_trace_verification_rate": row.get("logic_trace_verification_rate"),
                "cognitive_drift_rate": row.get("cognitive_drift_rate"),
                "self_correction_rate": row.get("self_correction_rate"),
                "tamper_rejection_rate": row.get("tamper_rejection_rate"),
                "average_grid_mask_rate": row.get("average_grid_mask_rate"),
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
                "the attack success rates reported by the paper.\n\n"
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
        report.write("| method | family | tier | size | quantization | status | ASR | semantic similarity | average watermark score | failures | comparison criterion |\n")
        report.write("|---|---|---|---:|---|---|---:|---:|---:|---:|---|\n")
        for row in comparison_rows:
            report.write(
                f"| {row['display_name']} | {row['model_family']} | {row.get('size_tier', 'N/A')} | "
                f"{row['parameter_size']} | {row['quantization']} | {row.get('run_status', 'completed')} | "
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

        report.write("## Cognitive Integrity Grid Baseline\n\n")
        if cognitive_row:
            report.write(
                f"- Attack success rate: {show(cognitive_row.get('attack_success_rate'))}\n"
            )
            report.write(
                f"- Semantic similarity: {show(cognitive_row.get('semantic_similarity'))}\n"
            )
            report.write(
                f"- Logic trace verification rate: {show(cognitive_row.get('logic_trace_verification_rate'))}\n"
            )
            report.write(
                f"- Cognitive drift rate: {show(cognitive_row.get('cognitive_drift_rate'))}\n"
            )
            report.write(
                f"- Self-correction rate: {show(cognitive_row.get('self_correction_rate'))}\n"
            )
            report.write(
                f"- Tampered trace rejection rate: {show(cognitive_row.get('tamper_rejection_rate'))}\n"
            )
            report.write(
                f"- Average grid mask rate: {show(cognitive_row.get('average_grid_mask_rate'))}\n\n"
            )
            report.write(
                "The grid artifact verifies an externally visible, deterministic token-state path. "
                "It does not verify the model's hidden chain-of-thought, prove that the response is "
                "safe, or turn the language model into a deterministic logic engine. Its watermark "
                "result should be interpreted as a grid-guided masking and rewrite baseline.\n\n"
            )
            if control_row:
                asr_difference = subtract_if_measured(
                    cognitive_row.get("attack_success_rate"),
                    control_row.get("attack_success_rate"),
                )
                similarity_difference = subtract_if_measured(
                    cognitive_row.get("semantic_similarity"),
                    control_row.get("semantic_similarity"),
                )
                report.write(f"- Normal rewrite control ASR: {show(control_row.get('attack_success_rate'))}\n")
                report.write(
                    f"- Grid ASR minus control ASR: {show(asr_difference)}\n"
                )
                report.write(
                    f"- Grid similarity minus control similarity: {show(similarity_difference)}\n\n"
                )
                if asr_difference is not None and asr_difference > 0:
                    report.write(
                        "The grid-masked rewrite removed the watermark more often than the normal "
                        "rewrite control in this run. Check the similarity difference before treating "
                        "that as a useful improvement.\n\n"
                    )
                elif asr_difference is not None:
                    report.write(
                        "The grid-masked rewrite did not outperform the normal rewrite control on "
                        "attack success in this run, so there is no evidence that the grid added a "
                        "watermark-attack benefit.\n\n"
                    )
                else:
                    report.write(
                        "The grid and normal rewrite outputs could not be compared because attack "
                        "success was not measured for both rows.\n\n"
                    )
        else:
            report.write("The Cognitive Integrity Grid baseline was not evaluated.\n\n")

        report.write("## Honest Conclusion\n\n")
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
