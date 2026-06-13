import argparse
import csv
import json
import os
import time

import torch
from sentence_transformers import SentenceTransformer

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
    prefixed_samples = 0

    for item in items:
        text = item.get(text_field, "")
        if not text:
            failed_samples += 1
            continue
        if text.startswith("student_id: 35571241"):
            prefixed_samples += 1

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
        "student_id_prefix_rate": prefixed_samples / len(items) if items else None,
        "runtime_seconds": time.time() - start_time,
    }


def evaluate_coda_artifacts(items):
    anchor_counts = []
    anchor_rates = []
    suspicious_counts = []

    for item in items:
        if isinstance(item.get("anchor_count"), (int, float)):
            anchor_counts.append(float(item["anchor_count"]))
        if isinstance(item.get("anchor_rate"), (int, float)):
            anchor_rates.append(float(item["anchor_rate"]))
        if isinstance(item.get("suspicious_token_count"), (int, float)):
            suspicious_counts.append(float(item["suspicious_token_count"]))

    return {
        "average_anchor_count": sum(anchor_counts) / len(anchor_counts) if anchor_counts else None,
        "average_anchor_rate": sum(anchor_rates) / len(anchor_rates) if anchor_rates else None,
        "average_suspicious_token_count": (
            sum(suspicious_counts) / len(suspicious_counts) if suspicious_counts else None
        ),
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
    parser.add_argument("--coda_models_config", default="")
    parser.add_argument("--spia_input", default="")
    parser.add_argument("--output_root", default="/content/sira_outputs")
    parser.add_argument("--similarity_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    with open(args.models_config, "r", encoding="utf-8-sig") as input_file:
        model_runs = json.load(input_file)
    coda_model_runs = []
    if args.coda_models_config and os.path.exists(args.coda_models_config):
        with open(args.coda_models_config, "r", encoding="utf-8-sig") as input_file:
            coda_model_runs = json.load(input_file)

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

    coda_records = []
    for model_run in coda_model_runs:
        coda_items = []
        if model_run.get("run_status") in {None, "completed"}:
            coda_items = read_jsonl(model_run["coda_path"], args.max_samples)
        if not coda_items:
            print(f"Recording missing CoDA output: {model_run['coda_path']}")
            coda_record = {
                **model_run,
                "label": f"coda_{model_run['label']}",
                "source_model_label": model_run["label"],
                "display_name": f"CoDA - {model_run['display_name']}",
                "method_type": "CoDA",
                "attack_path": model_run["coda_path"],
                "attack_success_rate": None,
                "average_watermark_score": None,
                "semantic_similarity": None,
                "number_of_samples": 0,
                "evaluated_samples": 0,
                "failed_samples": args.max_samples if args.max_samples > 0 else 0,
                "runtime_seconds": 0,
            }
            coda_records.append(coda_record)
            results.append(coda_record)
            continue

        print(f"Evaluating CoDA - {model_run['display_name']}: {len(coda_items)} samples")
        coda_result = evaluate_items(
            coda_items,
            "attack_text",
            watermark,
            similarity_model,
        )
        coda_artifacts = evaluate_coda_artifacts(coda_items)
        coda_record = {
            **model_run,
            "label": f"coda_{model_run['label']}",
            "source_model_label": model_run["label"],
            "display_name": f"CoDA - {model_run['display_name']}",
            "method_type": "CoDA",
            "attack_path": model_run["coda_path"],
            **coda_result,
            **coda_artifacts,
        }
        coda_records.append(coda_record)
        results.append(coda_record)

    spia_record = None
    spia_items = read_jsonl(args.spia_input, args.max_samples)
    if spia_items:
        print(f"Evaluating SPIA: {len(spia_items)} samples")
        spia_result = evaluate_items(
            spia_items,
            "attack_text",
            watermark,
            similarity_model,
        )
        spia_record = {
            "label": "spia",
            "display_name": "SPIA - Student-ID Prefix Injection Attack",
            "method_type": "SPIA",
            "model_family": "N/A",
            "model_name": "No rewrite model",
            "parameter_size": "N/A",
            "size_tier": "Proposed method",
            "quantization": "N/A",
            "attack_path": args.spia_input,
            **spia_result,
        }
        results.append(spia_record)

    results_dir = os.path.join(args.output_root, "results")
    save_json(os.path.join(results_dir, "transfer_eval.json"), results)
    save_csv(os.path.join(results_dir, "transfer_eval.csv"), results)
    if spia_record:
        save_json(os.path.join(results_dir, "spia_eval.json"), spia_record)
        save_csv(os.path.join(results_dir, "spia_eval.csv"), [spia_record])
    if coda_records:
        save_json(os.path.join(results_dir, "coda_eval.json"), coda_records)
        save_csv(os.path.join(results_dir, "coda_eval.csv"), coda_records)

    print(f"Saved transfer evaluation under: {results_dir}")


if __name__ == "__main__":
    main()
