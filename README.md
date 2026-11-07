# EC-417 P24 – Emotion in Tweets: Fine-Tuning DistilBERT

**One-line summary:** *<fill in at the end, e.g. "Fine-tuned DistilBERT reaches 0.9xx ± 0.00x macro-F1 on dair-ai/emotion (TF-IDF+LR: 0.8xx); class-weighted loss changes macro-F1 by +/-0.00x and rare-class (surprise) F1 by ...">*

Team: <Name 1> (<roll>), <Name 2> (<roll>) – NIT Andhra Pradesh, EC-417 (Oct 2026)

## 1. Problem
Classify a tweet into one of six emotions (sadness, joy, love, anger, fear, surprise). Why it matters: <1–2 sentences>.
**Metric:** macro-F1 (primary) + per-class F1, because classes are imbalanced and accuracy hides failures on rare classes such as *surprise*.

## 2. Data
dair-ai/emotion (Hugging Face), 16,000 / 2,000 / 2,000 official split kept unchanged. Class balance: <table/figure from EDA>. Licence/citation: see `data/README.md`.

## 3. Method
| Step | What |
|---|---|
| Trivial baseline | Predict the majority class |
| Baseline | TF-IDF (word 1–2-grams) + logistic regression; C/n-gram tuned on validation |
| Main model | `distilbert-base-uncased`, full fine-tuning with HF `Trainer` (lr 5e-5, 4 epochs, batch 32, max length 64, best epoch by validation macro-F1) |
| **Improvement (one change)** | Class-weighted cross-entropy (inverse training-frequency weights). Nothing else changes. |
| Optional | LoRA (r=16, q/v projections) vs full fine-tuning |

<diagram: data → tokenizer → DistilBERT → classifier head → (weighted) CE loss>

## 4. Results (test set, mean ± std over 3 seeds)
*Test numbers are computed once, after model selection on validation.*

| Model | Macro-F1 | Accuracy | Train time (s) | Trainable params |
|---|---|---|---|---|
| Trivial baseline | | | – | 0 |
| TF-IDF + LR | | | | |
| DistilBERT (full FT) | | | | |
| + class-weighted loss | | | | |

Figures: `results/figures/per_class_f1.png`, `results/figures/confusion_*.png`. Paste `results/summary.md` here.
Significance: bootstrap CI + McNemar in `results/extra_stats.json`.

## 5. Error analysis
Confusions (joy↔love, fear↔surprise, ...), 20 hand-analysed errors (`results/error_analysis_*.csv`), label-noise discussion.

## 6. Limitations and ethics
Distant-supervision label noise; English tweets only; possible train/test text overlap (see EDA); emotion inference can be misused for profiling/surveillance – this is an educational model, not for decisions about people; pre-trained model carries biases from its pre-training data.

## 7. Reproduce
```bash
pip install -r requirements.txt
bash run_all.sh            # or run notebooks/02_experiments.ipynb in Colab (T4)
```
Hardware: <GPU>, runtime: <min>. Python/library versions pinned in `requirements.txt`.

## 8. AI-use statement
<Tools used (e.g. Claude for scaffolding the repo/code structure), for what, and what we verified ourselves (we ran every experiment, read every line, checked metrics.csv against the README).>

## 9. References
- Saravia et al., CARER: Contextualized Affect Representations for Emotion Recognition, EMNLP 2018.
- Sanh et al., DistilBERT, a distilled version of BERT, 2019.
- Wolf et al., Transformers: State-of-the-Art NLP, EMNLP 2020 (demos).
- (if used) Hu et al., LoRA, 2021.
