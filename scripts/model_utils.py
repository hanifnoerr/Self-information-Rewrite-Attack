import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def choose_torch_dtype(dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(
    model_name,
    dtype_name="auto",
    load_in_4bit=False,
    load_in_8bit=False,
    loader_type="auto",
):
    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose either 4-bit or 8-bit loading, not both.")

    torch_dtype = choose_torch_dtype(dtype_name)
    model_args = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
    }

    if load_in_4bit:
        model_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        model_args["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading model: {model_name}")
    print(f"dtype={torch_dtype}, load_in_4bit={load_in_4bit}, load_in_8bit={load_in_8bit}")

    if loader_type == "auto":
        loader_type = "multimodal" if "gemma-4-" in model_name.lower() else "causal"

    if loader_type == "multimodal":
        from transformers import AutoModelForMultimodalLM

        print("loader_type=multimodal")
        model = AutoModelForMultimodalLM.from_pretrained(model_name, **model_args)
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_name)
    elif loader_type == "processor_causal":
        print("loader_type=processor_causal")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_name)
    else:
        print("loader_type=causal")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        tokenizer_or_processor = AutoTokenizer.from_pretrained(model_name)

    tokenizer = get_text_tokenizer(tokenizer_or_processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer_or_processor


def get_text_tokenizer(tokenizer_or_processor):
    if hasattr(tokenizer_or_processor, "tokenizer"):
        return tokenizer_or_processor.tokenizer
    return tokenizer_or_processor


def generate_chat_text(model, tokenizer_or_processor, messages, max_new_tokens=256):
    return generate_chat_texts(
        model,
        tokenizer_or_processor,
        [messages],
        max_new_tokens=max_new_tokens,
    )[0]


def generate_chat_texts(model, tokenizer_or_processor, message_batches, max_new_tokens=256):
    chat_texts = [
        tokenizer_or_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in message_batches
    ]

    if hasattr(tokenizer_or_processor, "tokenizer"):
        inputs = tokenizer_or_processor(
            text=chat_texts,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = tokenizer_or_processor(
            chat_texts,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    input_length = inputs["input_ids"].shape[-1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    response_ids = output_ids[:, input_length:]
    responses = get_text_tokenizer(tokenizer_or_processor).batch_decode(
        response_ids,
        skip_special_tokens=True,
    )

    return [response.strip() for response in responses]


def calculate_self_information_batch(model, tokenizer, texts):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        encoding = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    finally:
        tokenizer.padding_side = original_padding_side

    with torch.inference_mode():
        logits = model(**encoding).logits

    results = []
    for row_index in range(len(texts)):
        positions = torch.where(encoding["attention_mask"][row_index] == 1)[0]
        token_ids = encoding["input_ids"][row_index, positions]

        if len(token_ids) < 2:
            results.append((token_ids.tolist(), []))
            continue

        prediction_logits = logits[row_index, positions[:-1], :]
        target_ids = token_ids[1:]
        self_information = F.cross_entropy(
            prediction_logits.float(),
            target_ids,
            reduction="none",
        )
        results.append((token_ids.tolist(), self_information.tolist()))

    return results


def make_batches(items, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]
