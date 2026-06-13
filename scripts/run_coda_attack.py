import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from model_utils import generate_chat_text, get_text_tokenizer, load_model_and_tokenizer


ANCHOR_MARKER = "[ANCHOR]"


def build_coda_prompt(marked_text):
    return (
        "You are performing a Context-Anchor Desynchronization Attack (CoDA).\n"
        "Replace every [ANCHOR] marker with a short, natural connector or context phrase.\n"
        "Keep every visible word and fact unchanged whenever possible.\n"
        "Do not broadly paraphrase the paragraph.\n"
        "Make only the smallest grammatical edits required around each marker.\n"
        "Output only the completed paragraph.\n\n"
        f"Paragraph:\n{marked_text}"
    )


def calculate_self_information(model, tokenizer, text):
    with torch.no_grad():
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        outputs = model(**encoding)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        self_information = -torch.log(probabilities)

    input_ids = encoding["input_ids"][0]
    if len(input_ids) < 2:
        return input_ids.tolist(), []

    next_token_ids = input_ids[1:].unsqueeze(-1)
    values = self_information[:, :-1].gather(
        -1,
        next_token_ids.unsqueeze(0),
    ).squeeze().tolist()

    if isinstance(values, float):
        values = [values]
    return input_ids.tolist(), values


def select_anchor_indexes(token_ids, self_information_values, threshold):
    if not self_information_values:
        return [], []

    suspicious_cutoff = np.percentile(self_information_values, threshold)
    low_information_cutoff = np.percentile(self_information_values, 50)

    suspicious_indexes = []
    anchor_indexes = []

    # self_information_values[index] describes token_ids[index + 1].
    for value_index, target_value in enumerate(self_information_values):
        target_index = value_index + 1
        anchor_index = target_index - 1

        if target_value <= suspicious_cutoff:
            continue

        suspicious_indexes.append(target_index)

        # The first token has no measured self-information value, so skip it.
        if anchor_index == 0:
            continue

        anchor_value = self_information_values[anchor_index - 1]
        if anchor_value <= low_information_cutoff:
            anchor_indexes.append(anchor_index)

    anchor_indexes = sorted(set(anchor_indexes))

    # Always test at least one preceding anchor when a suspicious token exists.
    if not anchor_indexes and suspicious_indexes:
        fallback_targets = [target_index for target_index in suspicious_indexes if target_index > 1]
        if fallback_targets:
            anchor_indexes = [fallback_targets[0] - 1]

    return suspicious_indexes, anchor_indexes


def build_marked_text(token_ids, anchor_indexes, tokenizer):
    anchor_index_set = set(anchor_indexes)
    text_parts = []

    for token_index, token_id in enumerate(token_ids):
        if token_index in anchor_index_set:
            text_parts.append(f" {ANCHOR_MARKER} ")
        else:
            text_parts.append(
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )

    return "".join(text_parts).strip()


def main():
    parser = argparse.ArgumentParser(description="Run the CoDA anchor-desynchronization attack.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--loader_type",
        choices=["auto", "causal", "multimodal", "processor_causal"],
        default="auto",
    )
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer_or_processor = load_model_and_tokenizer(
        args.model_name,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        loader_type=args.loader_type,
    )
    tokenizer = get_text_tokenizer(tokenizer_or_processor)

    with open(args.input_path, "r", encoding="utf-8") as input_file:
        lines = [line for line in input_file if line.strip()]
    if args.max_samples > 0:
        lines = lines[:args.max_samples]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        for item_id, line in enumerate(tqdm(lines, desc="CoDA", unit="sample")):
            item = json.loads(line)
            watermarked_text = item["watermarked_text"]

            token_ids, self_information_values = calculate_self_information(
                model,
                tokenizer,
                watermarked_text,
            )
            suspicious_indexes, anchor_indexes = select_anchor_indexes(
                token_ids,
                self_information_values,
                args.threshold,
            )
            marked_text = build_marked_text(token_ids, anchor_indexes, tokenizer)
            suspicious_tokens = [
                tokenizer.decode([token_ids[index]], skip_special_tokens=True)
                for index in suspicious_indexes
            ]
            anchor_tokens = [
                tokenizer.decode([token_ids[index]], skip_special_tokens=True)
                for index in anchor_indexes
            ]

            messages = [{"role": "user", "content": build_coda_prompt(marked_text)}]
            generated_text = generate_chat_text(
                model,
                tokenizer_or_processor,
                messages,
                max_new_tokens=args.max_new_tokens,
            )

            result = {
                "id": item_id,
                "prompt": item.get("prompt", ""),
                "watermarked_text": watermarked_text,
                "unwatermarked_text": item.get("unwatermarked_text", ""),
                "marked_text": marked_text,
                "attack_text": generated_text,
                "suspicious_token_indexes": suspicious_indexes,
                "suspicious_tokens": suspicious_tokens,
                "anchor_token_indexes": anchor_indexes,
                "anchor_tokens": anchor_tokens,
                "suspicious_token_count": len(suspicious_indexes),
                "anchor_count": len(anchor_indexes),
                "anchor_rate": len(anchor_indexes) / len(token_ids) if token_ids else 0.0,
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"CoDA output: {args.output_path}")
    print(f"Runtime: {time.time() - start_time:.2f} seconds")
    print(f"Peak GPU memory: {peak_memory:.2f} GB")


if __name__ == "__main__":
    main()
