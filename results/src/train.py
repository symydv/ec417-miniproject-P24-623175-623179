"""One config, fixed seed, one command.

  python -m src.train --model majority
  python -m src.train --model tfidf_lr
  python -m src.train --model distilbert --seed 0                          # MAIN model
  python -m src.train --model distilbert --class_weighted --seed 0         # MAIN + IMPROVEMENT (weighted loss)
  python -m src.train --model distilbert --method lora --lr 5e-4 --epochs 6 --seed 0   # optional LoRA comparison

Each run appends rows to results/metrics.csv and saves test predictions to results/preds/.
Best epoch is chosen on VALIDATION macro-F1; the test set is predicted once, afterwards.
"""
import argparse, csv, math, os, random, time
import numpy as np
import pandas as pd

from src.data import load_emotion, label_names, as_lists, tokenize_dataset, MODEL_NAME, MAX_LENGTH
from src.models import build_majority, build_tfidf_lr, build_transformer, count_params, class_weights
from src.evaluate import compute_metrics

RES = "results"


def log_rows(name, seed, metrics):
    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/metrics.csv"; new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["model", "seed", "metric", "value"])
        for k, v in metrics.items(): w.writerow([name, seed, k, float(v)])


def save_preds(name, seed, texts, y_true, y_pred):
    os.makedirs(f"{RES}/preds", exist_ok=True)
    pd.DataFrame({"text": texts, "y_true": y_true, "y_pred": y_pred}).to_csv(f"{RES}/preds/{name}_seed{seed}.csv", index=False)


def run_sklearn(kind, seed):
    from sklearn.metrics import f1_score
    ds = load_emotion(); d = as_lists(ds)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = d["train"], d["validation"], d["test"]
    t0 = time.time()
    if kind == "majority":
        model = build_majority().fit(Xtr, ytr); name = "trivial_majority"; nparams = 0
    else:  # hyper-parameters tuned on VALIDATION only
        best, cfg = -1, None
        for C in (0.3, 1, 3, 10, 30, 100):
            for ng in ((1, 1), (1, 2)):
                s = f1_score(yva, build_tfidf_lr(C, ng).fit(Xtr, ytr).predict(Xva), average="macro")
                if s > best: best, cfg = s, (C, ng)
        print(f"best val macro-F1 {best:.4f} with C={cfg[0]} ngram={cfg[1]}")
        model = build_tfidf_lr(*cfg).fit(Xtr, ytr); name = "tfidf_lr"
        nparams = len(model.named_steps["tfidf"].vocabulary_) * 6   # one weight per (feature, class)
    train_time = time.time() - t0
    pred = model.predict(Xte)
    m = compute_metrics(yte, pred); m.update(train_time_s=train_time, trainable_params=nparams)
    log_rows(name, seed, m); save_preds(name, seed, Xte, yte, pred)
    print(name, {k: round(float(v), 4) for k, v in m.items()})


def run_transformer(a):
    import torch
    from torch import nn
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

    random.seed(a.seed); np.random.seed(a.seed); set_seed(a.seed)
    ds = load_emotion(); names = label_names(ds); id2label = dict(enumerate(names))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tds = tokenize_dataset(ds, tok, a.max_length)
    model = build_transformer(MODEL_NAME, len(names), id2label, a.method)
    trainable, total = count_params(model)
    print(f"trainable params: {trainable:,} / {total:,}")

    weights = None
    if a.class_weighted:   # THE improvement: weights from training-label frequencies
        weights = torch.tensor(class_weights(np.array(ds["train"]["label"]), len(names)), dtype=torch.float)
        print("class weights:", dict(zip(names, [round(x, 3) for x in weights.tolist()])))

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = weights.to(outputs.logits.device) if weights is not None else None
            loss = nn.CrossEntropyLoss(weight=w)(outputs.logits, labels)   # w=None -> plain CE (main model)
            return (loss, outputs) if return_outputs else loss

    def hf_metrics(p):
        m = compute_metrics(p.label_ids, p.predictions.argmax(-1))
        return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}

    name = "distilbert_lora" if a.method == "lora" else ("distilbert_full_weighted" if a.class_weighted else "distilbert_full")
    targs = TrainingArguments(
        output_dir=f"checkpoints/{name}_seed{a.seed}", num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=128, weight_decay=0.01, warmup_steps=int(0.06 * math.ceil(len(tds["train"]) / a.batch_size) * a.epochs),
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True,          # model selection on VALIDATION
        save_total_limit=1, fp16=torch.cuda.is_available(), seed=a.seed, report_to="none", logging_steps=100)
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=tds["train"], eval_dataset=tds["validation"],
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=hf_metrics)
    t0 = time.time(); trainer.train(); train_time = time.time() - t0

    val = trainer.evaluate(tds["validation"])
    out = trainer.predict(tds["test"])                                       # test touched once, after selection
    pred = out.predictions.argmax(-1)
    m = compute_metrics(out.label_ids, pred)
    m.update(train_time_s=train_time, trainable_params=trainable, total_params=total, val_macro_f1=val["eval_macro_f1"],
             peak_gpu_mem_gb=(torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0))
    log_rows(name, a.seed, m); save_preds(name, a.seed, ds["test"]["text"], out.label_ids, pred)
    print(name, a.seed, {k: round(float(v), 4) for k, v in m.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["majority", "tfidf_lr", "distilbert"], required=True)
    ap.add_argument("--method", choices=["full", "lora"], default="full")
    ap.add_argument("--class_weighted", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    a = ap.parse_args()
    run_sklearn(a.model, a.seed) if a.model in ("majority", "tfidf_lr") else run_transformer(a)
