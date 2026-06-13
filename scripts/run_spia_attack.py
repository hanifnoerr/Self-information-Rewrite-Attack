import argparse
import json
import os

from tqdm import tqdm


STUDENT_ID_PREFIX = "student_id: 35571241"


def main():
    parser = argparse.ArgumentParser(description="Run the Student-ID Prefix Injection Attack.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_samples", type=int, default=10)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as input_file:
        lines = [line for line in input_file if line.strip()]
    if args.max_samples > 0:
        lines = lines[:args.max_samples]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as output_file:
        for item_id, line in enumerate(tqdm(lines, desc="SPIA", unit="sample")):
            item = json.loads(line)
            result = {
                "id": item_id,
                "student_id": "35571241",
                "prompt": item.get("prompt", ""),
                "watermarked_text": item["watermarked_text"],
                "unwatermarked_text": item.get("unwatermarked_text", ""),
                "attack_text": f"{STUDENT_ID_PREFIX}\n{item['watermarked_text']}",
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"SPIA output: {args.output_path}")


if __name__ == "__main__":
    main()
