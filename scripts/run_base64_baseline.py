import argparse
import base64
import json
import os
import time

import torch
import transformers
from tqdm import tqdm

from model_utils import load_model_and_tokenizer


def read_jsonl(path, max_samples):
    items = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                items.append(json.loads(line))
            if max_samples > 0 and len(items) >= max_samples:
                break
    return items


def get_generated_text(outputs):
    generated = outputs[0]["generated_text"]
    if isinstance(generated, list):
        last_message = generated[-1]
        if isinstance(last_message, dict):
            return last_message.get("content", "")
        return str(last_message)
    return str(generated)


def write_raw_base64(items, raw_path, decoded_path, algorithm):
    with open(raw_path, "w", encoding="utf-8") as raw_file:
        with open(decoded_path, "w", encoding="utf-8") as decoded_file:
            for item_id, item in enumerate(items):
                watermarked_text = item["watermarked_text"]
                encoded_text = base64.b64encode(watermarked_text.encode("utf-8")).decode("ascii")

                raw_item = {
                    "id": item_id,
                    "algorithm": algorithm,
                    "prompt": item.get("prompt", ""),
                    "watermarked_text": watermarked_text,
                    "base64_output": encoded_text,
                    "decode_success": True,
                    "decode_error": "",
                }
                decoded_item = {
                    **raw_item,
                    "decoded_text": base64.b64decode(encoded_text).decode("utf-8"),
                    "decode_success": True,
                    "decode_error": "",
                }

                raw_file.write(json.dumps(raw_item, ensure_ascii=False) + "\n")
                decoded_file.write(json.dumps(decoded_item, ensure_ascii=False) + "\n")


def run_llm_baselines(items, args, llm_raw_path, normal_path, external_raw_path, external_decoded_path):
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

    with open(llm_raw_path, "w", encoding="utf-8") as llm_file:
        normal_file = open(normal_path, "w", encoding="utf-8") if args.run_normal_paraphrase else None
        external_raw_file = open(external_raw_path, "w", encoding="utf-8") if args.run_normal_paraphrase else None
        external_decoded_file = open(external_decoded_path, "w", encoding="utf-8") if args.run_normal_paraphrase else None
        try:
            for item_id, item in enumerate(tqdm(items, desc="LLM Base64 baseline")):
                watermarked_text = item["watermarked_text"]
                base64_prompt = (
                    "Paraphrase the following text while preserving the meaning.\n"
                    "Then output ONLY the Base64 encoding of the paraphrased text.\n"
                    "Do not output any explanation.\n"
                    "Do not include markdown.\n\n"
                    f"Text:\n{watermarked_text}\n\nBase64:"
                )
                messages = [{"role": "user", "content": base64_prompt}]
                outputs = pipeline(messages, max_new_tokens=args.base64_max_new_tokens, do_sample=False)

                llm_item = {
                    "id": item_id,
                    "algorithm": args.algorithm,
                    "model_name": args.model_name,
                    "prompt": item.get("prompt", ""),
                    "watermarked_text": watermarked_text,
                    "base64_output": get_generated_text(outputs),
                }
                llm_file.write(json.dumps(llm_item, ensure_ascii=False) + "\n")

                if normal_file:
                    normal_prompt = (
                        "Paraphrase the following text while preserving its meaning. "
                        "Output only the paraphrased text.\n\n"
                        f"Text:\n{watermarked_text}"
                    )
                    messages = [{"role": "user", "content": normal_prompt}]
                    outputs = pipeline(messages, max_new_tokens=args.paraphrase_max_new_tokens, do_sample=False)
                    paraphrase_text = get_generated_text(outputs)
                    normal_item = {
                        "id": item_id,
                        "algorithm": args.algorithm,
                        "model_name": args.model_name,
                        "prompt": item.get("prompt", ""),
                        "watermarked_text": watermarked_text,
                        "attack_text": paraphrase_text,
                    }
                    normal_file.write(json.dumps(normal_item, ensure_ascii=False) + "\n")

                    # This control isolates Base64 from paraphrasing. Encoding and
                    # decoding must return the exact same normal paraphrase.
                    encoded_paraphrase = base64.b64encode(paraphrase_text.encode("utf-8")).decode("ascii")
                    external_raw_item = {
                        "id": item_id,
                        "algorithm": args.algorithm,
                        "model_name": args.model_name,
                        "prompt": item.get("prompt", ""),
                        "watermarked_text": watermarked_text,
                        "base64_output": encoded_paraphrase,
                        "decode_success": True,
                        "decode_error": "",
                    }
                    external_decoded_item = {
                        **external_raw_item,
                        "decoded_text": base64.b64decode(encoded_paraphrase).decode("utf-8"),
                        "decoded_matches_normal_paraphrase": True,
                    }
                    external_raw_file.write(json.dumps(external_raw_item, ensure_ascii=False) + "\n")
                    external_decoded_file.write(json.dumps(external_decoded_item, ensure_ascii=False) + "\n")
        finally:
            if normal_file:
                normal_file.close()
            if external_raw_file:
                external_raw_file.close()
            if external_decoded_file:
                external_decoded_file.close()

    del pipeline, model, tokenizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Run raw and LLM-generated Base64 baselines.")
    parser.add_argument("--input_path", required=True, help="Official {algorithm}_response.json JSONL file")
    parser.add_argument("--output_root", default="/content/sira_outputs")
    parser.add_argument("--algorithm", default="KGW")
    parser.add_argument("--model_name", default="", help="Leave empty to skip the LLM Base64 baseline")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--run_normal_paraphrase", action="store_true")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--base64_max_new_tokens", type=int, default=2048)
    parser.add_argument("--paraphrase_max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    start_time = time.time()

    base64_dir = os.path.join(args.output_root, "base64")
    base64_llm_dir = os.path.join(args.output_root, "base64_llm")
    normal_dir = os.path.join(args.output_root, "normal_paraphrase")
    os.makedirs(base64_dir, exist_ok=True)
    os.makedirs(base64_llm_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    items = read_jsonl(args.input_path, args.max_samples)
    print(f"Loaded {len(items)} watermarked samples.")

    raw_path = os.path.join(base64_dir, "base64_raw.jsonl")
    decoded_path = os.path.join(base64_dir, "base64_decoded.jsonl")
    write_raw_base64(items, raw_path, decoded_path, args.algorithm)
    print(f"Saved raw Base64: {raw_path}")
    print(f"Saved decoded original text: {decoded_path}")

    if args.model_name:
        llm_raw_path = os.path.join(base64_llm_dir, "base64_llm_raw.jsonl")
        normal_path = os.path.join(normal_dir, "normal_paraphrase.jsonl")
        external_raw_path = os.path.join(base64_llm_dir, "normal_paraphrase_base64_raw.jsonl")
        external_decoded_path = os.path.join(base64_llm_dir, "normal_paraphrase_base64_decoded.jsonl")
        run_llm_baselines(
            items,
            args,
            llm_raw_path,
            normal_path,
            external_raw_path,
            external_decoded_path,
        )
        print(f"Saved LLM Base64 outputs: {llm_raw_path}")
        print(f"Saved externally encoded normal paraphrases: {external_raw_path}")

    print(f"Base64 baseline runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
