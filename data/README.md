# Data

- Dataset and version: dair-ai/emotion, configuration `split` (default) -- fill in the exact revision printed by `scripts/get_data.py`
- Official source URL: https://huggingface.co/datasets/dair-ai/emotion
- Access date: <YYYY-MM-DD>
- License / terms: <copy from the dataset card -- check it yourself and quote it here>
- Required citation: Saravia et al., "CARER: Contextualized Affect Representations for Emotion Recognition", EMNLP 2018.
- Files used: HF `datasets` cache (parquet files); nothing is committed to Git.
- Download/preparation command: `python scripts/get_data.py`
- Preprocessing and filtering: none beyond the model's own tokenizer (`distilbert-base-uncased` lower-cases). Max length 64 tokens (justified by the length distribution in notebooks/01_eda.ipynb).
- Label mapping: 0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise
- Train/validation/test split: official 16,000 / 2,000 / 2,000 (kept unchanged)
- Random seed(s): 0, 1, 2 (model training); the split itself is fixed by the dataset
- Known limitations, leakage risks, privacy concerns: labels come from distant supervision (hashtags) so they are noisy; English only; possible duplicate texts across splits (checked in notebooks/01_eda.ipynb); tweets are user-generated text -- do not redistribute or use for profiling individuals.

To reproduce:
1. `pip install -r requirements.txt`
2. `python scripts/get_data.py`
3. Run the commands in the project README (`bash run_all.sh`).
