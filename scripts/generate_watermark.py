import os
import gc
import json
import torch
import argparse
import time
from tqdm import tqdm
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from model_utils import load_model_and_tokenizer

def main(args):
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for algorithm in args.algorithms:
        print(f"Processing algorithm: {algorithm}")

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            args.model_path,
            dtype_name=args.dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )

        transformers_config = TransformersConfig(
            model=model,
            device=device,
            tokenizer=tokenizer,
            vocab_size=len(tokenizer),
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_length,
            do_sample=True,
            no_repeat_ngram_size=4
        )

        # Load watermark
        myWatermark = AutoWatermark.load(
            algorithm,
            algorithm_config=f'config/{algorithm}.json',
            transformers_config=transformers_config
        )

        # Input/output paths
        input_path = args.input_path
        output_path = os.path.join(args.output_dir, f"{algorithm}_response.json")

        with open(input_path, 'r') as f:
            lines = f.readlines()
        if args.max_samples > 0:
            lines = lines[:args.max_samples]

        # Determine how many lines already written
        if os.path.exists(output_path):
            with open(output_path, 'r') as out_f:
                existing_lines = sum(1 for _ in out_f)
        else:
            existing_lines = 0

        with open(output_path, 'a') as out_f:
            for i, line in enumerate(tqdm(lines, desc=f"Processing {algorithm}", unit="line")):
                if i < existing_lines:
                    continue

                item = json.loads(line)
                prompt = item.get('prefix', item.get('prompt'))
                if prompt is None:
                    raise KeyError("Input row must contain either 'prefix' or 'prompt'.")
                watermarked_text = myWatermark.generate_watermarked_text(prompt)
                unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)

                response_item = {
                    'prompt': prompt,
                    'watermarked_text': watermarked_text,
                    'unwatermarked_text': unwatermarked_text
                }
                out_f.write(json.dumps(response_item) + '\n')

        # Free memory
        del myWatermark, transformers_config, lines
        gc.collect()
        torch.cuda.empty_cache()

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"Watermark generation runtime: {time.time() - start_time:.2f} seconds")
    print(f"Watermark generation peak GPU memory: {peak_memory:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate watermarked/unwatermarked text with different algorithms.")

    parser.add_argument('--algorithms', nargs='+', default=['UPV', 'EWD'], help='List of watermarking algorithms to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained language model')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input .jsonl file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output JSON files')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='Which GPU(s) to make visible')
    parser.add_argument('--cuda_device', type=int, default=0, help='Which CUDA device to use (e.g., 0)')
    parser.add_argument('--dtype', choices=['auto', 'fp16', 'bf16'], default='auto')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--max_samples', type=int, default=0, help='0 means all input samples')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--min_length', type=int, default=230)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
