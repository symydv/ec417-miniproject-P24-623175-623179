"""Data loading. The official train/validation/test split is kept untouched."""

DATASET = "dair-ai/emotion"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64  # justified in notebooks/01_eda.ipynb (tweets are short)


def load_emotion():
    from datasets import load_dataset
    ds = load_dataset(DATASET)
    assert (len(ds["train"]), len(ds["validation"]), len(ds["test"])) == (16000, 2000, 2000), \
        "Unexpected split sizes - check the dataset card/config."
    return ds


def label_names(ds):
    return ds["train"].features["label"].names


def as_lists(ds):
    """(texts, labels) per split as plain lists - used by the sklearn baselines."""
    return {s: (list(ds[s]["text"]), list(ds[s]["label"])) for s in ("train", "validation", "test")}


def tokenize_dataset(ds, tokenizer, max_length=MAX_LENGTH):
    tok = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length), batched=True)
    tok = tok.rename_column("label", "labels")
    tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tok
