# EC-417 P24 – Emotion in Tweets: Fine-Tuning DistilBERT

**Fine-tuned DistilBERT reaches 0.891 ± 0.007 macro-F1 on dair-ai/emotion (TF-IDF + logistic regression: 0.831). A class-weighted loss leaves macro-F1 unchanged (0.895 ± 0.001) but trades precision for recall on the rare classes (surprise recall 0.74 → 0.95, precision 0.85 → 0.68 on seed 0).**

Team: TODO Name 1 (TODO roll), TODO Name 2 (TODO roll) – NIT Andhra Pradesh, EC-417 Introduction to Machine Intelligence (Oct 2026)

## 1. Problem
Classify an English tweet into one of six emotions: sadness, joy, love, anger, fear, surprise.
TODO (1–2 sentences on why emotion detection matters, e.g. customer feedback, mental-health monitoring research, with the ethics caveat in section 7).

**Metric: macro-F1** (primary), plus per-class F1/precision/recall. The classes are imbalanced (surprise is ≈3.6 % of the training data, joy ≈33.5 %), so accuracy hides failures on rare classes: predicting "joy" for every tweet already gives 0.347 accuracy but only 0.086 macro-F1.

## 2. Data
- **dair-ai/emotion** (Hugging Face), configuration `split`. Official split kept unchanged: 16,000 train / 2,000 validation / 2,000 test. Licence, citation, revision: see `data/README.md`.
- Class balance (training set, derived from class counts): sadness ≈29.1 %, joy ≈33.5 %, love ≈8.2 %, anger ≈13.5 %, fear ≈12.1 %, surprise ≈3.6 %. Test support: sadness 581, joy 695, love 159, anger 275, fear 224, surprise 66.
- EDA findings (`notebooks/01_eda.ipynb`):
  - Only 0.16 % of training tweets exceed 64 tokens, so `max_length = 64` is safe.
  - 31 duplicated texts inside train; 11 test texts also occur in train (5 train∩val, 3 val∩test); 30 training texts carry more than one label.
  - **All 11 train∩test texts have a different label in test than in train**, and the main model (seed 0) got 1 of 11 right. The overlap therefore does not inflate scores; it adds roughly 10 unavoidable errors (≈0.5 accuracy points).
  - Labels come from distant supervision (hashtags) and several sampled tweets are ambiguous or arguably mislabelled.

## 3. Method
| Step | What |
|---|---|
| Trivial baseline | Predict the majority class (joy) |
| Baseline | TF-IDF (word n-grams, `min_df=2`, sublinear tf) + logistic regression; C ∈ {0.3…100} and n-gram range chosen on validation macro-F1 |
| Main model | `distilbert-base-uncased` fine-tuned with the Hugging Face `Trainer`: lr 5e-5, 4 epochs, batch 32, max length 64, weight decay 0.01, 6 % warm-up, fp16; best epoch chosen on validation macro-F1 |
| **Improvement (one change)** | Class-weighted cross-entropy, w_c = N / (K · n_c) from training-label counts (sadness 0.57, joy 0.50, love 2.05, anger 1.24, fear 1.38, surprise 4.66). Nothing else changes. |

All models use the same split and metric. The test set is predicted once per run, after model selection on validation. Seeds 0, 1, 2 for the transformer runs.

TODO (optional: add a small diagram: tweet → tokenizer → DistilBERT → classifier head → (weighted) cross-entropy).

## 4. Results (test set)
Neural models: mean ± std over 3 seeds. Baselines are deterministic (one run; bootstrap CIs below).

| Model | Macro-F1 | Accuracy | Train time (s) | Trainable params |
|---|---|---|---|---|
| Trivial baseline (majority) | 0.086 | 0.347 | – | 0 |
| TF-IDF + LR | 0.831 | 0.881 | 101 † | 43,554 |
| DistilBERT (full fine-tuning) | 0.891 ± 0.007 | 0.931 ± 0.002 | 137 ± 7 | 66,958,086 |
| + class-weighted loss | 0.895 ± 0.001 | 0.930 ± 0.002 | 157 ± 38 ‡ | 66,958,086 |

† includes the 12-configuration hyper-parameter grid search on CPU. ‡ Run-to-run time differences come from Colab checkpoint-writing speed, not from the method. Peak GPU memory ≈1.37 GB (T4).

**Per-class F1** (mean ± std over seeds):

| Model | sadness | joy | love | anger | fear | surprise |
|---|---|---|---|---|---|---|
| Trivial | 0.000 | 0.516 | 0.000 | 0.000 | 0.000 | 0.000 |
| TF-IDF + LR | 0.917 | 0.912 | 0.755 | 0.870 | 0.847 | 0.683 |
| DistilBERT | 0.968 ± 0.000 | 0.949 ± 0.001 | 0.838 ± 0.012 | 0.926 ± 0.005 | 0.900 ± 0.014 | 0.761 ± 0.038 |
| + class weights | 0.969 ± 0.005 | 0.946 ± 0.002 | 0.845 ± 0.006 | 0.928 ± 0.008 | 0.894 ± 0.005 | 0.789 ± 0.004 |

**Uncertainty.** 95 % bootstrap CI of test macro-F1 (seed 0): TF-IDF [0.805, 0.852]; DistilBERT [0.872, 0.913]; + class weights [0.874, 0.912]. McNemar (seed 0, main vs weighted): the weighted model fixed 48 test tweets and broke 55, p = 0.55.

**Precision / recall on seed 0** (`results/classification_report_seed0.txt`):

| Class (support) | Precision main → weighted | Recall main → weighted |
|---|---|---|
| surprise (66) | 0.845 → 0.677 | 0.742 → 0.955 |
| love (159) | 0.807 → 0.766 | 0.843 → 0.950 |
| fear (224) | 0.884 → 0.907 | 0.915 → 0.871 |
| joy (695) | 0.952 → 0.980 | 0.945 → 0.917 |
| macro average | 0.898 → 0.872 | 0.891 → 0.927 |

Figures: `results/figures/per_class_f1.png`, `results/figures/confusion_*.png`, `results/figures/class_balance.png`, `results/figures/token_lengths.png`.

**Reading the results.**
1. DistilBERT beats TF-IDF by ≈0.06 macro-F1; the bootstrap intervals do not overlap.
2. The class-weighted loss does **not** improve macro-F1 (difference +0.004, smaller than the main model's seed std of 0.007; McNemar p = 0.55). It moves the operating point: macro recall rises 0.891 → 0.927 while macro precision falls 0.898 → 0.872, and the two effects cancel in F1.
3. It is preferable only if missing a rare emotion costs more than a false alarm. The lower variance on *surprise* (std 0.004 vs 0.038) is suggestive but rests on 3 seeds and 66 test tweets.

## 5. Error analysis
- Dominant confusions (pooled over seeds, row-normalised): love → joy (12 % main, 5 % weighted), surprise → fear (22 % main, 7 % weighted), and in the weighted model fear → surprise (8 %) and joy → love (7 %). TF-IDF is far worse on these pairs (love → joy 22 %, surprise → joy 14 %, surprise → fear 17 %).
- Label noise: the 11 duplicated train/test texts all have conflicting labels; 30 training texts have more than one label.
- TODO: summarise your hand analysis of 20 misclassified tweets (`results/error_analysis_distilbert_full_weighted.csv`): counts per category (label noise, ambiguity, negation, sarcasm, very short text) and 2–3 example tweets.

## 6. Limitations
- Noisy distant-supervision labels cap achievable performance; some errors are not model failures.
- Surprise has only 66 test tweets, so its per-class numbers are noisy (one tweet ≈ 1.5 recall points).
- Only 3 seeds; the baselines have no seed variance.
- One improvement tested; thresholds/softer class weights were not explored.
- Single dataset, English only, 2010s-era Twitter language.

## 7. Ethics
Emotion inference can be misused for profiling or surveillance; this model is an educational benchmark and must not be used to make decisions about individuals. Pre-trained language models carry biases from their pre-training data, and the label set (six emotions) is a coarse simplification. Tweets are user-generated text: do not redistribute or de-anonymise them.

## 8. Reproduce
```bash
git clone https://github.com/symydv/ec417-miniproject-P24-623175-623179.git
cd ec417-miniproject-P24-623175-623179
pip install -r requirements.txt
bash run_all.sh        # data download, 2 baselines, 3 seeds × 2 transformer configs, evaluation
```
Hardware: Google Colab free tier, NVIDIA T4. Runtime: ≈25–30 min for everything. Or run `notebooks/02_experiments.ipynb` top to bottom.

## 9. AI-use statement
TODO – be specific. Example: "We used Claude (Anthropic) to scaffold the repository structure and the training/evaluation code, and to help interpret results and draft wording. We ran every experiment ourselves on Colab, read and can explain every line of `src/`, checked every number in this README against `results/metrics.csv`, `results/summary.md` and `results/classification_report_seed0.txt`, and wrote the error analysis and conclusions ourselves."

## 10. References
- Saravia, Liu, Huang, Wu, Chen. *CARER: Contextualized Affect Representations for Emotion Recognition.* EMNLP 2018.
- Sanh, Debut, Chaumond, Wolf. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* 2019.
- Wolf et al. *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020 (System Demonstrations).
- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
