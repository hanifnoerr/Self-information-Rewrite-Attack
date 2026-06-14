import os
import json
import torch
import argparse
import time
from tqdm import tqdm
from model_utils import generate_chat_texts, load_model_and_tokenizer, make_batches


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
    parser.add_argument('--loader_type', choices=['auto', 'causal', 'multimodal', 'processor_causal'], default='auto')
    parser.add_argument('--batch_size', type=int, default=1)
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
        loader_type=args.loader_type,
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

        remaining_lines = lines[last_line:]
        with open(output_file, 'a') as out_f:
            progress = tqdm(total=len(remaining_lines), desc=f"Processing {algorithm}", unit="line")
            for line_batch in make_batches(remaining_lines, args.batch_size):
                items = [json.loads(line) for line in line_batch]
                message_batches = [
                    [{"role": "user", "content": fill_attack_prompt(item['ref_text'], item['blank_text'])}]
                    for item in items
                ]
                output_texts = generate_chat_texts(
                    model,
                    tokenizer,
                    message_batches,
                    max_new_tokens=256,
                )
                for item, output_text in zip(items, output_texts):
                    response_item = {
                        'prompt': item['prompt'],
                        'watermarked_text': item['watermarked_text'],
                        'unwatermarked_text': item['unwatermarked_text'],
                        'blank_text': item['blank_text'],
                        'ref_text': item['ref_text'],
                        'attack_text': output_text,
                    }
                    out_f.write(json.dumps(response_item) + '\n')
                out_f.flush()
                progress.update(len(items))
            progress.close()

    del model, tokenizer
    torch.cuda.empty_cache()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"Attack runtime: {time.time() - start_time:.2f} seconds")
    print(f"Attack peak GPU memory: {peak_memory:.2f} GB")


if __name__ == "__main__":
    args = parse_args()
    main(args)
