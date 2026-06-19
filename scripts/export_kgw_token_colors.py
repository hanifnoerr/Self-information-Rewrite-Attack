import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers import AutoTokenizer

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from watermark.kgw.kgw import KGWUtils


MODEL_LAYOUT = [
    {
        "label": "llama_3_8b",
        "name": "Llama 3 8B Instruct",
    },
    {
        "label": "qwen_2_5_7b",
        "name": "Qwen 2.5 7B Instruct",
    },
    {
        "label": "mistral_7b_v0_3",
        "name": "Mistral 7B Instruct v0.3",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export exact KGW token colors for selected SIRA samples."
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--output_path",
        default="",
        help="Defaults to <output_root>/visualization/kgw_token_colors.json",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        default=[0, 100, 200, 300, 400],
    )
    parser.add_argument(
        "--tokenizer_name",
        default="facebook/opt-1.3b",
    )
    parser.add_argument(
        "--required_gpu_substring",
        default="A100",
        help="Use an empty value to permit any CUDA GPU.",
    )
    parser.add_argument(
        "--allow_environment_mismatch",
        action="store_true",
        help="Permit GPU or package versions that differ from environment.json.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def validate_runtime(required_gpu_substring):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required because KGW token partitions differ across "
            "CPU and CUDA random-number implementations."
        )

    gpu_name = torch.cuda.get_device_name(0)
    if (
        required_gpu_substring
        and required_gpu_substring.lower() not in gpu_name.lower()
    ):
        raise RuntimeError(
            f"Expected a GPU containing '{required_gpu_substring}', "
            f"but found '{gpu_name}'."
        )
    return gpu_name


def validate_recorded_environment(
    output_root,
    gpu_name,
    allow_environment_mismatch,
):
    environment_path = output_root / "environment.json"
    if not environment_path.exists():
        print("No environment.json found; runtime version matching was skipped.")
        return

    recorded = read_json(environment_path)
    checks = {
        "GPU": (recorded.get("gpu"), gpu_name),
        "PyTorch": (recorded.get("torch_version"), torch.__version__),
        "Transformers": (
            recorded.get("transformers_version"),
            transformers.__version__,
        ),
    }
    mismatches = []
    for name, (expected, actual) in checks.items():
        if expected and expected != actual:
            mismatches.append(
                f"{name}: recorded '{expected}', current '{actual}'"
            )

    if mismatches and not allow_environment_mismatch:
        mismatch_text = "\n".join(f"- {item}" for item in mismatches)
        raise RuntimeError(
            "The current runtime does not match the recorded experiment:\n"
            f"{mismatch_text}\n"
            "Use the recorded environment or pass "
            "--allow_environment_mismatch for an explicitly non-identical "
            "export."
        )
    if mismatches:
        print("Warning: exporting with environment differences:")
        for item in mismatches:
            print(f"- {item}")
    else:
        print("Runtime matches the recorded experiment environment.")


def validate_sample_ids(sample_ids, row_count):
    for sample_id in sample_ids:
        if sample_id < 0 or sample_id >= row_count:
            raise ValueError(
                f"Sample ID {sample_id} is outside the available range "
                f"0 to {row_count - 1}."
            )


def build_kgw_utils(tokenizer, config, device):
    kgw_config = SimpleNamespace(
        gamma=config["gamma"],
        delta=config["delta"],
        hash_key=config["hash_key"],
        z_threshold=config["z_threshold"],
        prefix_length=config["prefix_length"],
        f_scheme=config["f_scheme"],
        window_scheme=config["window_scheme"],
        vocab_size=len(tokenizer),
        device=device,
    )
    return KGWUtils(kgw_config), kgw_config


def classify_text(text, prompt, tokenizer, kgw_utils, kgw_config, device):
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].to(device)
    score, flags = kgw_utils.score_sequence(token_ids)

    prompt_token_count = 0
    if prompt and text.startswith(prompt):
        prompt_token_count = len(
            tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
        )

    tokens = []
    for index, (token_id, flag) in enumerate(
        zip(token_ids.tolist(), flags)
    ):
        tokens.append(
            {
                "text": tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                ),
                "class": (
                    "green"
                    if flag == 1
                    else "red"
                    if flag == 0
                    else "neutral"
                ),
                "prompt": index < prompt_token_count,
            }
        )

    scored_count = len(flags) - kgw_config.prefix_length
    green_count = sum(flag == 1 for flag in flags)
    return {
        "score": float(score),
        "is_watermarked": bool(score > kgw_config.z_threshold),
        "green_count": green_count,
        "red_count": scored_count - green_count,
        "green_ratio": (
            green_count / scored_count
            if scored_count > 0
            else None
        ),
        "tokens": tokens,
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path
        else output_root / "visualization" / "kgw_token_colors.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_name = validate_runtime(args.required_gpu_substring)
    validate_recorded_environment(
        output_root,
        gpu_name,
        args.allow_environment_mismatch,
    )
    device = torch.device("cuda:0")
    print(f"GPU: {gpu_name}")
    print(f"Torch: {torch.__version__}")
    print(f"Tokenizer: {args.tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    repo_dir = Path(__file__).resolve().parents[1]
    config = read_json(repo_dir / "config" / "KGW.json")
    kgw_utils, kgw_config = build_kgw_utils(
        tokenizer,
        config,
        device,
    )

    watermarked_rows = read_jsonl(
        output_root / "watermarked" / "KGW_response.json"
    )
    validate_sample_ids(args.sample_ids, len(watermarked_rows))

    attack_rows = {}
    for model in MODEL_LAYOUT:
        attack_rows[model["label"]] = read_jsonl(
            output_root
            / "sira_models"
            / model["label"]
            / "final"
            / "KGW_attack.json"
        )

    examples = {}
    for position, sample_id in enumerate(args.sample_ids, start=1):
        print(
            f"[{position}/{len(args.sample_ids)}] "
            f"Exporting sample {sample_id}"
        )
        source = watermarked_rows[sample_id]
        prompt = source.get("prompt", "")
        watermarked_text = source["watermarked_text"]
        example = {
            "watermarked": classify_text(
                watermarked_text,
                prompt,
                tokenizer,
                kgw_utils,
                kgw_config,
                device,
            ),
            "models": {},
        }

        for model in MODEL_LAYOUT:
            attack_row = attack_rows[model["label"]][sample_id]
            if attack_row["watermarked_text"] != watermarked_text:
                raise ValueError(
                    f"Input mismatch for {model['label']}, "
                    f"sample {sample_id}."
                )
            example["models"][model["label"]] = classify_text(
                attack_row["attack_text"],
                "",
                tokenizer,
                kgw_utils,
                kgw_config,
                device,
            )
        examples[str(sample_id)] = example

    payload = {
        "metadata": {
            "algorithm": "KGW",
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "tokenizer": args.tokenizer_name,
            "gamma": config["gamma"],
            "delta": config["delta"],
            "z_threshold": config["z_threshold"],
            "hash_key": config["hash_key"],
            "prefix_length": config["prefix_length"],
            "f_scheme": config["f_scheme"],
            "window_scheme": config["window_scheme"],
        },
        "sample_ids": args.sample_ids,
        "examples": examples,
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)

    print(f"Saved exact KGW token colors: {output_path}")


if __name__ == "__main__":
    main()
