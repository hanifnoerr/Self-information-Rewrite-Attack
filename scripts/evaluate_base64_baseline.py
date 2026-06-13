import argparse
import csv
import json
import math
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
        text = item.get(text_field, "")
        original = item.get("watermarked_text", "")
        if text and original:
            pairs.append((original, text))

    if not pairs:
        return None

    originals = [pair[0] for pair in pairs]
    outputs = [pair[1] for pair in pairs]
    original_embeddings = similarity_model.encode(originals, normalize_embeddings=True)
    output_embeddings = similarity_model.encode(outputs, normalize_embeddings=True)
    scores = (original_embeddings * output_embeddings).sum(axis=1)
    return float(scores.mean())


def calculate_perplexity(text, model, tokenizer):
    if not text.strip():
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return float(math.exp(min(loss.item(), 20)))


def evaluate_items(name, detector_input, items, watermark, similarity_model, model, tokenizer, calculate_ppl):
    start_time = time.time()
    scores = []
    watermarked_flags = []
    perplexities = []
    failed_samples = 0

    for item in items:
        text = item.get(detector_input, "")
        if not text:
            failed_samples += 1
            continue

        try:
            result = watermark.detect_watermark(text, return_dict=True)
            scores.append(float(result["score"]))
            watermarked_flags.append(bool(result["is_watermarked"]))
            if calculate_ppl and "raw" not in name:
                perplexity = calculate_perplexity(text, model, tokenizer)
                if perplexity is not None:
                    perplexities.append(perplexity)
        except Exception as error:
            failed_samples += 1
            print(f"[{name}] Evaluation failed for one sample: {error}")

    evaluated_samples = len(watermarked_flags)
    attack_success_rate = None
    if evaluated_samples:
        attack_success_rate = sum(not flag for flag in watermarked_flags) / evaluated_samples

    decode_values = [item.get("decode_success") for item in items if "decode_success" in item]
    decode_success_rate = None
    if decode_values:
        decode_success_rate = sum(bool(value) for value in decode_values) / len(decode_values)

    semantic_similarity = None
    if "raw" not in name:
        semantic_similarity = calculate_similarity(items, detector_input, similarity_model)

    return {
        "method": name,
        "detector_input": detector_input,
        "watermark_algorithm": items[0].get("algorithm", "") if items else "",
        "attack_success_rate": attack_success_rate,
        "average_watermark_score": sum(scores) / len(scores) if scores else None,
        "semantic_similarity": semantic_similarity,
        "decode_success_rate": decode_success_rate,
        "average_perplexity": sum(perplexities) / len(perplexities) if perplexities else None,
        "number_of_samples": len(items),
        "evaluated_samples": evaluated_samples,
        "failed_samples": failed_samples,
        "runtime_seconds": time.time() - start_time,
        "gpu_peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0,
        "note": "Raw Base64 is not readable natural language." if "raw" in name else "",
    }


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["method"]
    with open(path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SIRA and Base64 outputs with an official watermark detector.")
    parser.add_argument("--generation_model", default="facebook/opt-1.3b")
    parser.add_argument("--algorithm", default="KGW")
    parser.add_argument("--watermarked_input", default="")
    parser.add_argument("--sira_tiny_input", default="")
    parser.add_argument("--sira_small_input", default="")
    parser.add_argument("--base64_raw_input", default="")
    parser.add_argument("--base64_decoded_input", default="")
    parser.add_argument("--base64_llm_raw_input", default="")
    parser.add_argument("--base64_llm_decoded_input", default="")
    parser.add_argument("--normal_paraphrase_input", default="")
    parser.add_argument("--output_root", default="/content/sira_outputs")
    parser.add_argument("--similarity_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--calculate_ppl", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.generation_model,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
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

    sources = [
        ("watermarked_original", "watermarked_text", args.watermarked_input),
        ("SIRA-Tiny", "attack_text", args.sira_tiny_input),
        ("SIRA-Small", "attack_text", args.sira_small_input),
        ("Base64 raw encoding", "base64_output", args.base64_raw_input),
        ("Base64 decoded original text", "decoded_text", args.base64_decoded_input),
        ("LLM Base64 paraphrase raw string", "base64_output", args.base64_llm_raw_input),
        ("LLM Base64 paraphrase decoded text", "decoded_text", args.base64_llm_decoded_input),
        ("normal paraphrase baseline", "attack_text", args.normal_paraphrase_input),
    ]

    results = []
    for name, text_field, path in sources:
        items = read_jsonl(path, args.max_samples)
        for item in items:
            item.setdefault("algorithm", args.algorithm)
        if items:
            print(f"Evaluating {name}: {len(items)} samples")
            results.append(
                evaluate_items(
                    name,
                    text_field,
                    items,
                    watermark,
                    similarity_model,
                    model,
                    tokenizer,
                    args.calculate_ppl,
                )
            )

    results_dir = os.path.join(args.output_root, "results")
    save_json(os.path.join(results_dir, "all_eval.json"), results)
    save_csv(os.path.join(results_dir, "all_eval.csv"), results)

    for result in results:
        if result["method"] == "SIRA-Tiny":
            save_json(os.path.join(results_dir, "sira_tiny_eval.json"), result)
            save_csv(os.path.join(results_dir, "sira_tiny_eval.csv"), [result])
        elif result["method"] == "SIRA-Small":
            save_json(os.path.join(results_dir, "sira_small_eval.json"), result)
            save_csv(os.path.join(results_dir, "sira_small_eval.csv"), [result])

    base64_llm_results = [result for result in results if result["method"].startswith("LLM Base64")]
    if base64_llm_results:
        base64_llm_dir = os.path.join(args.output_root, "base64_llm")
        save_json(os.path.join(base64_llm_dir, "base64_llm_eval.json"), base64_llm_results)
        save_csv(os.path.join(base64_llm_dir, "base64_llm_eval.csv"), base64_llm_results)

    print(f"Saved evaluation files under: {args.output_root}")


if __name__ == "__main__":
    main()
