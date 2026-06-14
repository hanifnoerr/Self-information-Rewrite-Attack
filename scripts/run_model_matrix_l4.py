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
    parser = argparse.ArgumentParser(description="Run the configured SIRA attack models.")
    parser.add_argument("--config_path", default="config/model_matrix_l4.json")
    parser.add_argument("--repo_dir", default="/content/Self-information-Rewrite-Attack")
    parser.add_argument("--output_root", default="/content/sira_outputs")
    parser.add_argument("--algorithm", default="KGW")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as input_file:
        model_runs = json.load(input_file)

    models_config_path = os.path.join(args.output_root, "model_runs.json")
    for model_run in model_runs:
        model_run["attack_path"] = os.path.join(
            args.output_root,
            "sira_models",
            model_run["label"],
            "final",
            f"{args.algorithm}_attack.json",
        )
        model_run["run_status"] = "checking_access"
        model_run["run_error"] = ""
        model_run["batch_size"] = args.batch_size
        model_run["batch_size_note"] = "Requested batch size; resumed lines may use an earlier setting."
    save_json(models_config_path, model_runs)

    for model_run in model_runs:
        model_name = model_run["model_name"]
        print("\n" + "=" * 80)
        print(f"Checking and running: {model_run['display_name']}")
        print("=" * 80)

        completed_samples = count_nonempty_lines(model_run["attack_path"])
        if completed_samples >= args.samples:
            model_run["access_status"] = "not_rechecked"
            model_run["run_status"] = "completed"
            print(f"Skipping completed SIRA output with {completed_samples} samples.")
            save_json(models_config_path, model_runs)
            continue

        try:
            hf_hub_download(model_name, "config.json")
            model_run["access_status"] = "available"
        except Exception as error:
            model_run["access_status"] = "unavailable"
            model_run["run_status"] = "skipped_access"
            model_run["run_error"] = str(error)
            print(f"Skipping inaccessible model: {model_name}")
            print(error)
            save_json(models_config_path, model_runs)
            continue

        environment = os.environ.copy()
        environment["MODEL_NAME"] = model_name
        environment["MODEL_LABEL"] = model_run["label"]
        environment["LOAD_IN_4BIT"] = "true" if model_run["load_in_4bit"] else "false"
        environment["LOADER_TYPE"] = model_run["loader_type"]
        environment["ALGORITHM"] = args.algorithm
        environment["SAMPLES"] = str(args.samples)
        environment["BATCH_SIZE"] = str(args.batch_size)
        environment["OUTPUT_ROOT"] = args.output_root

        model_run["run_status"] = "running"
        save_json(models_config_path, model_runs)

        try:
            subprocess.run(
                ["bash", "scripts/run_sira_model_l4.sh"],
                cwd=args.repo_dir,
                env=environment,
                check=True,
            )
            completed_samples = count_nonempty_lines(model_run["attack_path"])
            if completed_samples >= args.samples:
                model_run["run_status"] = "completed"
            else:
                model_run["run_status"] = "failed"
                model_run["run_error"] = (
                    f"Runner finished with {completed_samples} of {args.samples} attack samples."
                )
        except subprocess.CalledProcessError as error:
            model_run["run_status"] = "failed"
            log_path = os.path.join(
                args.output_root,
                "logs",
                f"sira_{model_run['label']}.log",
            )
            model_run["run_error"] = (
                f"Runner exited with status {error.returncode}. Log: {log_path}"
            )
            print(f"Model run failed but the matrix will continue: {model_name}")

        save_json(models_config_path, model_runs)

    print(f"Saved model matrix status: {models_config_path}")


if __name__ == "__main__":
    main()
