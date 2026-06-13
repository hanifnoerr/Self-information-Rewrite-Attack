import argparse
import csv
import json
import os
import time

import torch
from sentence_transformers import SentenceTransformer

from cognitive_integrity import BehavioralVerifier, CognitiveDriftError
from model_utils import load_model_and_tokenizer
from utils.transformers_config import TransformersConfig
from watermark.auto_watermark import AutoWatermark


def read_jsonl(path, max_samples):
    if not path or not os.path.exists(path):
        return []

    items = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                items.append(json.loads(line))
            if max_samples > 0 and len(items) >= max_samples:
                break
    return items


def calculate_similarity(items, text_field, similarity_model):
    pairs = []
    for item in items:
        original = item.get("watermarked_text", "")
        output = item.get(text_field, "")
        if original and output:
            pairs.append((original, output))

    if not pairs:
        return None

    original_embeddings = similarity_model.encode(
        [pair[0] for pair in pairs],
        normalize_embeddings=True,
    )
    output_embeddings = similarity_model.encode(
        [pair[1] for pair in pairs],
        normalize_embeddings=True,
    )
    scores = (original_embeddings * output_embeddings).sum(axis=1)
    return float(scores.mean())


def evaluate_items(items, text_field, watermark, similarity_model):
    start_time = time.time()
    scores = []
    watermarked_flags = []
    failed_samples = 0

    for item in items:
        text = item.get(text_field, "")
        if not text:
            failed_samples += 1
            continue

        try:
            result = watermark.detect_watermark(text, return_dict=True)
            scores.append(float(result["score"]))
            watermarked_flags.append(bool(result["is_watermarked"]))
        except Exception as error:
            failed_samples += 1
            print(f"Evaluation failed for one sample: {error}")

    evaluated_samples = len(watermarked_flags)
    attack_success_rate = None
    if items:
        attack_success_rate = sum(not flag for flag in watermarked_flags) / len(items)

    return {
        "attack_success_rate": attack_success_rate,
        "average_watermark_score": sum(scores) / len(scores) if scores else None,
        "semantic_similarity": calculate_similarity(items, text_field, similarity_model),
        "number_of_samples": len(items),
        "evaluated_samples": evaluated_samples,
        "failed_samples": failed_samples,
        "runtime_seconds": time.time() - start_time,
    }


def evaluate_cognitive_artifacts(items, grid_size):
    verifier = BehavioralVerifier(grid_size=grid_size)
    verified_traces = 0
    drift_samples = 0
    corrected_samples = 0
    rejected_tampered_traces = 0
    mask_rates = []

    for item in items:
        try:
            verifier.verify_logic_trace(item.get("LOGIC_TRACE"))
            verified_traces += 1
        except CognitiveDriftError:
            pass

        tampered_trace = {
            "grid_path": [[0, 0], [grid_size - 1, grid_size - 1]],
            "safety_verification": "PASSED",
            "reasoning_integrity_check": "Verified against constraint matrix",
        }
        try:
            verifier.verify_logic_trace(tampered_trace)
        except CognitiveDriftError:
            rejected_tampered_traces += 1

        if item.get("cognitive_drift_detected"):
            drift_samples += 1
        if item.get("self_correction_performed"):
            corrected_samples += 1
        if isinstance(item.get("mask_rate"), (int, float)):
            mask_rates.append(float(item["mask_rate"]))

    sample_count = len(items)
    return {
        "logic_trace_verification_rate": verified_traces / sample_count if sample_count else None,
        "cognitive_drift_rate": drift_samples / sample_count if sample_count else None,
        "self_correction_rate": corrected_samples / sample_count if sample_count else None,
        "tamper_rejection_rate": rejected_tampered_traces / sample_count if sample_count else None,
        "average_grid_mask_rate": sum(mask_rates) / len(mask_rates) if mask_rates else None,
    }


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Evaluate SIRA transfer across attack-model families.")
    parser.add_argument("--generation_model", default="facebook/opt-1.3b")
    parser.add_argument("--algorithm", default="KGW")
    parser.add_argument("--watermarked_input", required=True)
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--cognitive_input", default="")
    parser.add_argument("--cognitive_model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--cognitive_quantization", default="bf16")
    parser.add_argument("--cognitive_grid_size", type=int, default=2)
    parser.add_argument("--output_root", default="/content/sira_outputs")
    parser.add_argument("--similarity_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    with open(args.models_config, "r", encoding="utf-8-sig") as input_file:
        model_runs = json.load(input_file)

    model, tokenizer = load_model_and_tokenizer(args.generation_model, dtype_name=args.dtype)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        device=device,
    )
    watermark = AutoWatermark.load(
        args.algorithm,
        algorithm_config=f"config/{args.algorithm}.json",
        transformers_config=transformers_config,
    )
    similarity_model = SentenceTransformer(args.similarity_model, device=device)

    results = []
    original_items = read_jsonl(args.watermarked_input, args.max_samples)
    original_result = evaluate_items(original_items, "watermarked_text", watermark, similarity_model)
    results.append(
        {
            "label": "watermarked_original",
            "display_name": "Watermarked original",
            "model_family": "N/A",
            "model_name": args.generation_model,
            "parameter_size": "N/A",
            "quantization": "N/A",
            **original_result,
        }
    )

    for model_run in model_runs:
        attack_items = []
        if model_run.get("run_status") in {None, "completed"}:
            attack_items = read_jsonl(model_run["attack_path"], args.max_samples)
        if not attack_items:
            print(f"Recording missing output: {model_run['attack_path']}")
            results.append(
                {
                    "method_type": "SIRA",
                    **model_run,
                    "attack_success_rate": None,
                    "average_watermark_score": None,
                    "semantic_similarity": None,
                    "number_of_samples": 0,
                    "evaluated_samples": 0,
                    "failed_samples": args.max_samples if args.max_samples > 0 else 0,
                    "runtime_seconds": 0,
                }
            )
            continue

        print(f"Evaluating {model_run['display_name']}: {len(attack_items)} samples")
        result = evaluate_items(attack_items, "attack_text", watermark, similarity_model)
        results.append({"method_type": "SIRA", **model_run, **result})

    cognitive_record = None
    control_record = None
    cognitive_items = read_jsonl(args.cognitive_input, args.max_samples)
    if cognitive_items:
        print(f"Evaluating Cognitive Integrity Grid Masking: {len(cognitive_items)} samples")
        cognitive_result = evaluate_items(
            cognitive_items,
            "attack_text",
            watermark,
            similarity_model,
        )
        cognitive_artifacts = evaluate_cognitive_artifacts(
            cognitive_items,
            args.cognitive_grid_size,
        )
        cognitive_record = {
            "label": "cognitive_integrity_grid",
            "display_name": "Cognitive Integrity Grid Masking",
            "method_type": "Grid-guided masking baseline",
            "model_family": "Llama rewrite model",
            "model_name": args.cognitive_model,
            "parameter_size": "3B",
            "quantization": args.cognitive_quantization,
            "attack_path": args.cognitive_input,
            **cognitive_result,
            **cognitive_artifacts,
        }
        results.append(cognitive_record)

        control_result = evaluate_items(
            cognitive_items,
            "control_text",
            watermark,
            similarity_model,
        )
        control_record = {
            "label": "normal_rewrite_control",
            "display_name": "Normal Rewrite Control",
            "method_type": "Rewrite control",
            "model_family": "Llama rewrite model",
            "model_name": args.cognitive_model,
            "parameter_size": "3B",
            "quantization": args.cognitive_quantization,
            "attack_path": args.cognitive_input,
            **control_result,
        }
        results.append(control_record)

    results_dir = os.path.join(args.output_root, "results")
    save_json(os.path.join(results_dir, "transfer_eval.json"), results)
    save_csv(os.path.join(results_dir, "transfer_eval.csv"), results)
    if cognitive_record:
        save_json(os.path.join(results_dir, "cognitive_integrity_eval.json"), cognitive_record)
        save_csv(os.path.join(results_dir, "cognitive_integrity_eval.csv"), [cognitive_record])
    if control_record:
        save_json(os.path.join(results_dir, "normal_rewrite_control_eval.json"), control_record)
        save_csv(os.path.join(results_dir, "normal_rewrite_control_eval.csv"), [control_record])

    print(f"Saved transfer evaluation under: {results_dir}")


if __name__ == "__main__":
    main()
