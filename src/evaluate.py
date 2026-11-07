"""Metrics + plots. `python -m src.evaluate` builds summary tables and figures from results/."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
RES = "results"
ORDER = ["trivial_majority", "tfidf_lr", "distilbert_full", "distilbert_full_weighted", "distilbert_lora"]
MAIN, IMPROVED = "distilbert_full", "distilbert_full_weighted"


def compute_metrics(y_true, y_pred):
    """dict with accuracy, macro_f1 and f1_<class> for each class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, average="macro", labels=list(range(6)), zero_division=0)}
    per = f1_score(y_true, y_pred, average=None, labels=list(range(6)), zero_division=0)
    for n, v in zip(LABELS, per):
        out[f"f1_{n}"] = v
    return out


def _ms(s):
    s = s.dropna()
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def summary_tables():
    m = pd.read_csv(f"{RES}/metrics.csv")
    wide = m.pivot_table(index=["model", "seed"], columns="metric", values="value").reset_index()
    models = [x for x in ORDER if x in set(wide.model)]
    rows, per_class = [], []
    for name in models:
        w = wide[wide.model == name]
        rows.append({"Model": name, "Macro-F1": _ms(w["macro_f1"]), "Accuracy": _ms(w["accuracy"]),
                     "Train time (s)": _ms(w["train_time_s"]),
                     "Trainable params": int(w["trainable_params"].iloc[0]), "n_seeds": len(w)})
        per_class.append({"Model": name, **{c: _ms(w[f"f1_{c}"]) for c in LABELS}})
    t1, t2 = pd.DataFrame(rows), pd.DataFrame(per_class)
    with open(f"{RES}/summary.md", "w") as f:
        f.write("## Overall (test set, mean ± std over seeds)\n\n" + t1.to_markdown(index=False) +
                "\n\n## Per-class F1 (test set)\n\n" + t2.to_markdown(index=False) + "\n")
    print(t1.to_string(index=False)); print(t2.to_string(index=False))
    return wide


def load_preds(name, seed):
    return pd.read_csv(f"{RES}/preds/{name}_seed{seed}.csv")


def confusion_plot(name, seeds):
    cm = sum(confusion_matrix(load_preds(name, s).y_true, load_preds(name, s).y_pred, labels=list(range(6))) for s in seeds)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6), LABELS, rotation=45); ax.set_yticks(range(6), LABELS)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{name}: row-normalised confusion")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{RES}/figures/confusion_{name}.png", dpi=150); plt.close(fig)


def per_class_plot(wide, models):
    fig, ax = plt.subplots(figsize=(8, 4)); width = 0.8 / len(models)
    for k, name in enumerate(models):
        w = wide[wide.model == name]
        means = [w[f"f1_{c}"].mean() for c in LABELS]
        errs = [w[f"f1_{c}"].std(ddof=1) if len(w) > 1 else 0 for c in LABELS]
        ax.bar(np.arange(6) + k * width, means, width, yerr=errs, label=name, capsize=2)
    ax.set_xticks(np.arange(6) + width * (len(models) - 1) / 2, LABELS)
    ax.set_ylabel("test F1"); ax.legend(fontsize=7); ax.set_title("Per-class F1 (mean ± std over seeds)")
    fig.tight_layout(); fig.savefig(f"{RES}/figures/per_class_f1.png", dpi=150); plt.close(fig)


def bootstrap_ci(name, seed=0, B=1000):
    """95% bootstrap CI of test macro-F1 (the only uncertainty available for deterministic baselines)."""
    p = load_preds(name, seed); rng = np.random.default_rng(0); n = len(p); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v.append(f1_score(p.y_true.values[i], p.y_pred.values[i], average="macro", zero_division=0))
    return np.percentile(v, [2.5, 97.5])


def mcnemar(a, b, seed=0):
    """Exact McNemar test between two models on the same test set."""
    from scipy.stats import binomtest
    pa, pb = load_preds(a, seed), load_preds(b, seed)
    ca, cb = pa.y_true == pa.y_pred, pb.y_true == pb.y_pred
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return {"a": a, "b": b, "a_wrong_b_right": n01, "a_right_b_wrong": n10, "p_value": p}


def error_analysis(name, seed=0, n=20):
    p = load_preds(name, seed); w = p[p.y_true != p.y_pred].sample(n=n, random_state=0).copy()
    w["true"] = w.y_true.map(dict(enumerate(LABELS))); w["pred"] = w.y_pred.map(dict(enumerate(LABELS)))
    w["your_explanation"] = ""   # <- YOU fill this in by hand (label noise? sarcasm? negation? ambiguous?)
    w[["text", "true", "pred", "your_explanation"]].to_csv(f"{RES}/error_analysis_{name}.csv", index=False)


def main():
    wide = summary_tables()
    models = [m for m in ORDER if m in set(wide.model)]
    per_class_plot(wide, models)
    extra = {}
    for name in models:
        seeds = sorted(int(s) for s in wide[wide.model == name].seed)
        confusion_plot(name, seeds)
        lo, hi = bootstrap_ci(name, seeds[0]); extra[f"{name}_bootstrap95_macroF1"] = [round(lo, 4), round(hi, 4)]
    if MAIN in models and IMPROVED in models:
        extra["mcnemar_main_vs_improved"] = mcnemar(MAIN, IMPROVED)
        error_analysis(IMPROVED)
    json.dump(extra, open(f"{RES}/extra_stats.json", "w"), indent=2, default=float)
    print(json.dumps(extra, indent=2, default=float))


if __name__ == "__main__":
    main()
