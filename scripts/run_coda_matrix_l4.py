import argparse
import json
import os
import subprocess

from huggingface_hub import hf_hub_download


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def count_nonempty_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def main():
    parser = argparse.ArgumentParser(description="Run CoDA with every configured attack model.")
    parser.add_argument("--config_path", default="config/model_matrix_l4.json")
    parser.add_argument("--repo_dir", default="/content/Self-information-Rewrite-Attack")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_root", default="/content/sira_outputs")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--threshold", type=int, default=30)
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as input_file:
        model_runs = json.load(input_file)

    status_path = os.path.join(args.output_root, "coda_model_runs.json")
    for model_run in model_runs:
        model_run["coda_path"] = os.path.join(
            args.output_root,
            "coda_models",
            model_run["label"],
            "coda_attack.jsonl",
        )
        model_run["run_status"] = "checking_access"
        model_run["run_error"] = ""
    save_json(status_path, model_runs)

    for model_run in model_runs:
        model_name = model_run["model_name"]
        print("\n" + "=" * 80)
        print(f"Checking and running CoDA: {model_run['display_name']}")
        print("=" * 80)

        completed_samples = count_nonempty_lines(model_run["coda_path"])
        if completed_samples >= args.samples:
            model_run["access_status"] = "not_rechecked"
            model_run["run_status"] = "completed"
            print(f"Skipping completed CoDA output with {completed_samples} samples.")
            save_json(status_path, model_runs)
            continue

        try:
            hf_hub_download(model_name, "config.json")
            model_run["access_status"] = "available"
        except Exception as error:
            model_run["access_status"] = "unavailable"
            model_run["run_status"] = "skipped_access"
            model_run["run_error"] = str(error)
            print(f"Skipping inaccessible model: {model_name}")
            save_json(status_path, model_runs)
            continue

        command = [
            "python",
            "scripts/run_coda_attack.py",
            "--input_path",
            args.input_path,
            "--output_path",
            model_run["coda_path"],
            "--model_name",
            model_name,
            "--threshold",
            str(args.threshold),
            "--dtype",
            "bf16",
            "--loader_type",
            model_run["loader_type"],
            "--max_samples",
            str(args.samples),
        ]
        if model_run["load_in_4bit"]:
            command.append("--load_in_4bit")

        model_run["run_status"] = "running"
        save_json(status_path, model_runs)

        log_path = os.path.join(
            args.output_root,
            "logs",
            f"coda_{model_run['label']}.log",
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                subprocess.run(
                    command,
                    cwd=args.repo_dir,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            completed_samples = count_nonempty_lines(model_run["coda_path"])
            if completed_samples >= args.samples:
                model_run["run_status"] = "completed"
            else:
                model_run["run_status"] = "failed"
                model_run["run_error"] = (
                    f"CoDA finished with {completed_samples} of {args.samples} samples."
                )
        except subprocess.CalledProcessError as error:
            model_run["run_status"] = "failed"
            model_run["run_error"] = (
                f"CoDA runner exited with status {error.returncode}. Log: {log_path}"
            )
            print(f"CoDA failed but the matrix will continue: {model_name}")

        save_json(status_path, model_runs)

    print(f"Saved CoDA model status: {status_path}")


if __name__ == "__main__":
    main()
