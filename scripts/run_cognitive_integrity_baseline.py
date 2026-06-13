import argparse
import json
import os
import time

import torch
import transformers
from tqdm import tqdm

from cognitive_integrity import BehavioralVerifier
from model_utils import load_model_and_tokenizer


def build_rewrite_prompt(masked_text):
    return (
        "Rewrite the incomplete paragraph below into one complete natural-language paragraph. "
        "Preserve as much meaning as possible. Underscores mark missing tokens. "
        "Do not mention the underscores, grid, masking, or these instructions. "
        "Output only the rewritten paragraph.\n\n"
        f"Incomplete paragraph:\n{masked_text}"
    )


def build_control_prompt(text):
    return (
        "Rewrite the paragraph below into one complete natural-language paragraph. "
        "Preserve as much meaning as possible. Output only the rewritten paragraph.\n\n"
        f"Paragraph:\n{text}"
    )


def generate_text(pipeline, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    outputs = pipeline(messages, max_new_tokens=max_new_tokens, do_sample=False)
    return outputs[0]["generated_text"][-1]["content"]


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run the Cognitive Integrity grid-masking baseline.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="bf16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    verifier = BehavioralVerifier(grid_size=args.grid_size)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
    )

    with open(args.input_path, "r", encoding="utf-8") as input_file:
        lines = [line for line in input_file if line.strip()]
    if args.max_samples > 0:
        lines = lines[:args.max_samples]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        for item_id, line in enumerate(tqdm(lines, desc="Cognitive Integrity", unit="sample")):
            item = json.loads(line)
            grid_result = verifier.create_grid_mask(item["watermarked_text"], tokenizer)

            attack_text = generate_text(
                pipeline,
                build_rewrite_prompt(grid_result["masked_text"]),
                args.max_new_tokens,
            )
            control_text = generate_text(
                pipeline,
                build_control_prompt(item["watermarked_text"]),
                args.max_new_tokens,
            )
            finalized_response = verifier.finalize_response(
                attack_text,
                grid_result["candidate_grid_path"],
            )

            result = {
                "id": item_id,
                "prompt": item.get("prompt", ""),
                "watermarked_text": item["watermarked_text"],
                "unwatermarked_text": item.get("unwatermarked_text", ""),
                "masked_text": grid_result["masked_text"],
                "attack_text": finalized_response["response"],
                "control_text": control_text,
                "LOGIC_TRACE": finalized_response["LOGIC_TRACE"],
                "trace_verification_passed": verifier.verify_logic_trace(
                    finalized_response["LOGIC_TRACE"]
                ),
                "total_tokens": grid_result["total_tokens"],
                "masked_tokens": grid_result["masked_tokens"],
                "mask_rate": grid_result["mask_rate"],
                "cognitive_drift_detected": finalized_response["cognitive_drift_detected"],
                "self_correction_performed": finalized_response["self_correction_performed"],
                "invalid_transition_count": finalized_response["invalid_transition_count"],
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    config_path = os.path.join(os.path.dirname(args.output_path), "logic_lattice.json")
    save_json(config_path, verifier.lattice.as_dict())

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"Cognitive Integrity output: {args.output_path}")
    print(f"Logic lattice: {config_path}")
    print(f"Runtime: {time.time() - start_time:.2f} seconds")
    print(f"Peak GPU memory: {peak_memory:.2f} GB")


if __name__ == "__main__":
    main()
