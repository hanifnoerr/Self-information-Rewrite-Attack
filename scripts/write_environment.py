import argparse
import json
import os
import platform
import subprocess

import torch
import transformers


def run_command(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"unavailable: {error}"


def main():
    parser = argparse.ArgumentParser(description="Save the Colab experiment environment.")
    parser.add_argument("--output_path", default="/content/sira_outputs/environment.json")
    parser.add_argument("--tiny_model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--small_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--generation_model", default="facebook/opt-1.3b")
    parser.add_argument("--tiny_quantization", default="bf16")
    parser.add_argument("--small_quantization", default="bf16")
    parser.add_argument("--models_config", default="")
    args = parser.parse_args()

    attack_models = []
    if args.models_config and os.path.exists(args.models_config):
        with open(args.models_config, "r", encoding="utf-8") as input_file:
            attack_models = json.load(input_file)

    data = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU",
        "gpu_memory_gb": (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if torch.cuda.is_available()
            else 0
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "nvidia_smi": run_command(["nvidia-smi"]),
        "models": {
            "generation_model": args.generation_model,
            "sira_tiny": args.tiny_model,
            "sira_small": args.small_model,
        },
        "quantization": {
            "sira_tiny": args.tiny_quantization,
            "sira_small": args.small_quantization,
        },
        "attack_models": attack_models,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)

    print(json.dumps(data, indent=2))
    print(f"Saved environment details to: {args.output_path}")


if __name__ == "__main__":
    main()
