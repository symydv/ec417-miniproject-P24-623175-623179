"""Baselines, main model, and the improvement options."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_majority():
    return DummyClassifier(strategy="most_frequent")


def build_tfidf_lr(C=1.0, ngram=(1, 2)):
    # vectoriser inside the Pipeline -> fitted on training text only
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def build_transformer(model_name, num_labels, id2label, method="full"):
    """'full' = fine-tune everything (main model). 'lora' = parameter-efficient variant (optional stretch)."""
    from transformers import AutoModelForSequenceClassification
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
    if method == "lora":
        from peft import LoraConfig, get_peft_model, TaskType
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
                         target_modules=["q_lin", "v_lin"])  # DistilBERT attention projections
        model = get_peft_model(model, cfg)  # SEQ_CLS keeps the classification head trainable
    return model


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def class_weights(labels, num_labels):
    """Inverse-frequency weights from TRAINING labels only: w_c = N / (K * n_c)."""
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    return counts.sum() / (num_labels * counts)
