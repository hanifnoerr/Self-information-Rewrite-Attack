import os
import json
import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
from model_utils import (
    calculate_self_information_batch,
    generate_chat_texts,
    get_text_tokenizer,
    load_model_and_tokenizer,
    make_batches,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run paraphrasing + self-information blanking pipeline.")
    parser.add_argument('--input_dir', type=str, default='/root/Self-information-Rewrite-Attack-main/dataset/c4/watermarked/',
                        help='Directory containing {algorithm}_response.json input files')
    parser.add_argument('--result_dir', type=str, default='/root/Self-information-Rewrite-Attack-main/dataset/c4/watermarked/',
                        help='Directory to store ref and blank output files')
    parser.add_argument('--model_path', type=str, default='/root/models/meta-llama/Llama-3.2-3B-Instruct/')  
    parser.add_argument('--threshold', type=int, default=30)  
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs to use, e.g., "0,1,2"')
    parser.add_argument('--algorithms', type=str, default='KGW',
                        help='Comma-separated list of watermark algorithms to process')
    parser.add_argument('--dtype', choices=['auto', 'fp16', 'bf16'], default='auto')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--loader_type', choices=['auto', 'causal', 'multimodal', 'processor_causal'], default='auto')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=0, help='0 means all input samples')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def fill_parapharseprompt(input_text):
    return (
        "You are a paraphraser. You are given an input passage 'INPUT'. "
        "You should paraphrase 'INPUT' to print 'OUTPUT'. 'OUTPUT' should be diverse and different "
        "as much as possible from 'INPUT' and should not copy any part verbatim from 'INPUT'. "
        "However, 'OUTPUT' should preserve the information in the INPUT. "
        "You should print 'OUTPUT' and nothing else so that it is easy for me to parse.\nINPUT: "
        + input_text
    )

if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()
    algorithms = args.algorithms.split(",")

    print(f"PID: {os.getpid()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Stage 1: Paraphrasing
    print("=== Stage 1: Generate reference text ===")
    paraphrase_model, paraphrase_tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        loader_type=args.loader_type,
    )
    
    for algorithm in algorithms:
        input_file = os.path.join(args.input_dir, f'{algorithm}_response.json')
        ref_output_file = os.path.join(args.result_dir, f'{algorithm}_ref.json')

        with open(input_file, 'r') as f:
            lines = f.readlines()
        if args.max_samples > 0:
            lines = lines[:args.max_samples]

        last_line = 0
        if os.path.exists(ref_output_file):
            with open(ref_output_file, 'r') as f:
                last_line = sum(1 for _ in f)

        remaining_lines = lines[last_line:]
        with open(ref_output_file, 'a') as out_f:
            progress = tqdm(total=len(remaining_lines), desc=f"Stage 1: {algorithm}", unit="line")
            for line_batch in make_batches(remaining_lines, args.batch_size):
                items = [json.loads(line) for line in line_batch]
                message_batches = [
                    [
                        {"role": "system", "content": "You are a helpful rewriter."},
                        {"role": "user", "content": fill_parapharseprompt(item['watermarked_text'])},
                    ]
                    for item in items
                ]
                output_texts = generate_chat_texts(
                    paraphrase_model,
                    paraphrase_tokenizer,
                    message_batches,
                    max_new_tokens=256,
                )
                for item, output_text in zip(items, output_texts):
                    response_item = {
                        'prompt': item['prompt'],
                        'watermarked_text': item['watermarked_text'],
                        'unwatermarked_text': item['unwatermarked_text'],
                        'ref_text': output_text,
                    }
                    out_f.write(json.dumps(response_item) + '\n')
                out_f.flush()
                progress.update(len(items))
            progress.close()

    del paraphrase_model, paraphrase_tokenizer
    torch.cuda.empty_cache()

    # Stage 2: Self-Information Blanking
    print("=== Stage 2: Self-Information Blanking ===")

    class SelfInformationCalculator:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self._prepare_model()
        
        def _prepare_model(self):
            self.model.eval()
            print('Model and tokenizer loaded successfully.')

        def transform_tokens(self, tokens, self_info_values, threshold_low):
            percentile = np.percentile(self_info_values, threshold_low)
            transformed_tokens = []
            temp_tokens = []

            for token, self_info in zip(tokens, self_info_values):
                if self_info > percentile:
                    if temp_tokens:
                        transformed_tokens.append(f"({' '.join(temp_tokens)})")
                        temp_tokens = []
                    transformed_tokens.append('_')
                else:
                    if temp_tokens:
                        transformed_tokens.append(f"({' '.join(temp_tokens)})")
                        temp_tokens = []
                    transformed_tokens.append(token)

            if temp_tokens:
                transformed_tokens.append(f"({' '.join(temp_tokens)})")

            return transformed_tokens

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype_name=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        loader_type=args.loader_type,
    )
    calculator = SelfInformationCalculator(
        model=model,
        tokenizer=get_text_tokenizer(tokenizer),
    )
    threshold = args.threshold

    for algorithm in algorithms:
        ref_output_file = os.path.join(args.result_dir, f'{algorithm}_ref.json')
        blank_output_file = os.path.join(args.result_dir, f'{algorithm}_blank.json')

        processed_lines = 0
        if os.path.exists(blank_output_file):
            with open(blank_output_file, 'r') as f:
                processed_lines = sum(1 for _ in f)
            print(f"Found {processed_lines} processed lines for {algorithm}")

        with open(ref_output_file, 'r') as f:
            lines = f.readlines()
        if args.max_samples > 0:
            lines = lines[:args.max_samples]

        remaining_lines = lines[processed_lines:]
        with open(blank_output_file, 'a') as out_f:
            progress = tqdm(total=len(remaining_lines), desc=f"Stage 2: {algorithm}", unit="line")
            for line_batch in make_batches(remaining_lines, args.batch_size):
                items = [json.loads(line) for line in line_batch]
                text_results = calculate_self_information_batch(
                    model,
                    calculator.tokenizer,
                    [item['watermarked_text'] for item in items],
                )
                for item, (token_ids, self_info_values) in zip(items, text_results):
                    tokens = [
                        calculator.tokenizer.decode(token_id)
                        for token_id in token_ids[1:]
                    ]
                    transformed_tokens = calculator.transform_tokens(
                        tokens,
                        self_info_values,
                        threshold,
                    )
                    response_item = {
                        'prompt': item['prompt'],
                        'watermarked_text': item['watermarked_text'],
                        'unwatermarked_text': item['unwatermarked_text'],
                        'ref_text': item['ref_text'],
                        'blank_text': "".join(transformed_tokens),
                    }
                    out_f.write(json.dumps(response_item) + '\n')
                out_f.flush()
                progress.update(len(items))
            progress.close()

    del model, tokenizer, calculator
    torch.cuda.empty_cache()
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"Pre-attack runtime: {time.time() - start_time:.2f} seconds")
    print(f"Pre-attack peak GPU memory: {peak_memory:.2f} GB")
