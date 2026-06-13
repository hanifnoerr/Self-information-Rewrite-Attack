import argparse
import csv
import json
import os


def read_by_id(path):
    if not path or not os.path.exists(path):
        return {}

    items = {}
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                item = json.loads(line)
                items[item["id"]] = item
    return items


def main():
    parser = argparse.ArgumentParser(description="Create a per-sample side-by-side Base64 comparison.")
    parser.add_argument("--raw_original", required=True)
    parser.add_argument("--decoded_original", required=True)
    parser.add_argument("--llm_raw", required=True)
    parser.add_argument("--llm_decoded", required=True)
    parser.add_argument("--normal_paraphrase", required=True)
    parser.add_argument("--external_raw", required=True)
    parser.add_argument("--external_decoded", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    raw_original = read_by_id(args.raw_original)
    decoded_original = read_by_id(args.decoded_original)
    llm_raw = read_by_id(args.llm_raw)
    llm_decoded = read_by_id(args.llm_decoded)
    normal_paraphrase = read_by_id(args.normal_paraphrase)
    external_raw = read_by_id(args.external_raw)
    external_decoded = read_by_id(args.external_decoded)

    all_ids = sorted(
        set(raw_original)
        | set(decoded_original)
        | set(llm_raw)
        | set(llm_decoded)
        | set(normal_paraphrase)
        | set(external_raw)
        | set(external_decoded)
    )

    rows = []
    for item_id in all_ids:
        source = raw_original.get(item_id, {})
        direct_decoded = llm_decoded.get(item_id, {})
        normal = normal_paraphrase.get(item_id, {})
        external = external_decoded.get(item_id, {})

        rows.append(
            {
                "id": item_id,
                "watermarked_text": source.get("watermarked_text", ""),
                "raw_original_base64": source.get("base64_output", ""),
                "decoded_original_text": decoded_original.get(item_id, {}).get("decoded_text", ""),
                "decoded_original_matches_watermarked": (
                    decoded_original.get(item_id, {}).get("decoded_text", "")
                    == source.get("watermarked_text", "")
                ),
                "direct_llm_base64_output": llm_raw.get(item_id, {}).get("base64_output", ""),
                "direct_llm_decoded_text": direct_decoded.get("decoded_text", ""),
                "direct_llm_decode_success": direct_decoded.get("decode_success", False),
                "direct_llm_decode_error": direct_decoded.get("decode_error", ""),
                "normal_paraphrase": normal.get("attack_text", ""),
                "normal_paraphrase_external_base64": external_raw.get(item_id, {}).get("base64_output", ""),
                "normal_paraphrase_external_decoded": external.get("decoded_text", ""),
                "external_decoded_matches_normal_paraphrase": (
                    external.get("decoded_text", "") == normal.get("attack_text", "")
                ),
            }
        )

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.output_csv, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} side-by-side rows to: {args.output_jsonl}")
    print(f"Saved side-by-side CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
