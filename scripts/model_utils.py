import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def choose_torch_dtype(dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(model_name, dtype_name="auto", load_in_4bit=False, load_in_8bit=False):
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

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
