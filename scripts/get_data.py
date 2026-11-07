"""Download dair-ai/emotion and print the facts you must record in data/README.md."""
from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("dair-ai/emotion")  # default config: 16k / 2k / 2k
    print(ds)
    print("Labels:", ds["train"].features["label"].names)
    info = ds["train"].info
    print("Dataset version:", info.version, "| config:", info.config_name, "| license field:", info.license)
    print("-> copy these into data/README.md, plus today's date.")
