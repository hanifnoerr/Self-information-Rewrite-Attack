import os
import json
import torch
import argparse
import time
import transformers
from tqdm import tqdm
from model_utils import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run paraphrasing attack against blanked input using a specified model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the HuggingFace model.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing {algorithm}_blank.json files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write {algorithm}_attack.json files.')
    parser.add_argument('--algorithms', type=str, default='KGW,Unigram,UPV,DIP,EWD,EXP,SIR',
                        help='Comma-separated list of watermark algorithms to process.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs to use, e.g., "0,1,2"')
    parser.add_argument('--dtype', choices=['auto', 'fp16', 'bf16'], default='auto')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0, help='0 means all input samples')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def fill_attack_prompt(reference_text, blank_text):
    prompt = (
        "You will be shown one reference paragraph and one incomplete paragraph.\n"
        "Your task is to write a complete paragraph using incomplete paragraph.\n"
        "The complete paragraph should have similar length with reference paragraph.\n"
        "You need to include all the information in the reference. \n"
        "But do not take the expression and words in the reference paragraph.\n"
        "You should only answer the complete paragraph.\n"
        f"reference: {reference_text}\n"
        f"incomplete pragraph: {blank_text}\n"
    )
    return prompt


def main(args):
    start_time = time.time()
    print(f"PID: {os.getpid()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load model and build generation pipeline
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False
    )

    algorithms = args.algorithms.split(",")

    for algorithm in algorithms:
        input_file = os.path.join(args.input_dir, f'{algorithm}_blank.json')
        output_file = os.path.join(args.output_dir, f'{algorithm}_attack.json')

        if not os.path.exists(input_file):
            print(f"[!] Skipping {algorithm}: input file not found → {input_file}")
            continue

        with open(input_file, 'r') as f:
            lines = f.readlines()
        if args.max_samples > 0:
            lines = lines[:args.max_samples]

        last_line = 0
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                last_line = sum(1 for _ in f)

        with open(output_file, 'a') as out_f:
            for line in tqdm(lines[last_line:], desc=f"Processing {algorithm}", unit="line"):
                item = json.loads(line)
                prompt = item['prompt']
                watermarked_text = item['watermarked_text']
                unwatermarked_text = item['unwatermarked_text']
                blank_text = item['blank_text']
                ref_text = item['ref_text']

                input_text = fill_attack_prompt(ref_text, blank_text)
                messages = [{"role": "user", "content": input_text}]
                outputs = pipeline(messages, max_new_tokens=256, do_sample=False)

                output_text = outputs[0]["generated_text"][-1]["content"]

                response_item = {
                    'prompt': prompt,
                    'watermarked_text': watermarked_text,
                    'unwatermarked_text': unwatermarked_text,
                    'blank_text': blank_text,
                    'ref_text': ref_text,
                    'attack_text': output_text,
                }
                out_f.write(json.dumps(response_item) + '\n')

    del pipeline, model, tokenizer
    torch.cuda.empty_cache()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"Attack runtime: {time.time() - start_time:.2f} seconds")
    print(f"Attack peak GPU memory: {peak_memory:.2f} GB")


if __name__ == "__main__":
    args = parse_args()
    main(args)
