import argparse
import base64
import json
import re


def clean_base64_text(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"^\s*(base64|output)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = "".join(cleaned.split())

    matches = re.findall(r"[A-Za-z0-9+/_=-]{8,}", cleaned)
    if matches:
        cleaned = max(matches, key=len)

    return cleaned


def decode_base64_text(text):
    cleaned = clean_base64_text(text)
    padding = (-len(cleaned)) % 4
    cleaned += "=" * padding

    try:
        decoded_bytes = base64.b64decode(cleaned, validate=True)
    except Exception:
        decoded_bytes = base64.urlsafe_b64decode(cleaned)

    return decoded_bytes.decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Decode Base64 strings from a JSONL file.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    total = 0
    success_count = 0

    with open(args.input_path, "r", encoding="utf-8") as input_file:
        with open(args.output_path, "w", encoding="utf-8") as output_file:
            for line in input_file:
                if not line.strip():
                    continue

                item = json.loads(line)
                total += 1

                result = {
                    "id": item.get("id", total - 1),
                    "watermarked_text": item.get("watermarked_text", ""),
                    "base64_output": item.get("base64_output", ""),
                    "decoded_text": "",
                    "decode_success": False,
                    "decode_error": "",
                }

                for key in ["prompt", "algorithm", "model_name"]:
                    if key in item:
                        result[key] = item[key]

                try:
                    result["decoded_text"] = decode_base64_text(result["base64_output"])
                    result["decode_success"] = True
                    success_count += 1
                except Exception as error:
                    result["decode_error"] = str(error)

                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Decoded {success_count}/{total} Base64 outputs.")
    print(f"Saved decoded JSONL to: {args.output_path}")


if __name__ == "__main__":
    main()
