import os
import sys

from huggingface_hub import HfApi, login


def main():
    token = os.environ.get("HF_TOKEN", "")

    if not token:
        try:
            from google.colab import userdata

            token = userdata.get("HF_TOKEN")
        except Exception as error:
            print("Hugging Face token was not found in the VM environment or Colab Secrets.")
            print("Add a Colab Secret named HF_TOKEN and grant the notebook access.")
            print(f"Colab Secrets error: {error}")
            sys.exit(2)

    login(token=token, add_to_git_credential=False)
    user = HfApi().whoami(token=token)
    print(f"Hugging Face login is ready for: {user.get('name', 'unknown user')}")

    models = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]
    for model_name in models:
        try:
            HfApi().model_info(model_name, token=token)
            print(f"Model access OK: {model_name}")
        except Exception as error:
            print(f"Model access failed: {model_name}")
            print("Accept the Meta Llama license on Hugging Face or replace the model name.")
            print(error)
            sys.exit(3)


if __name__ == "__main__":
    main()
