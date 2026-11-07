#!/usr/bin/env bash
# Regenerates every number in the README. ~25-40 min on a free Colab/Kaggle T4.
set -e
python scripts/get_data.py
rm -f results/metrics.csv
python -m src.train --model majority
python -m src.train --model tfidf_lr
for s in 0 1 2; do
  python -m src.train --model distilbert --seed $s                       # main model
  python -m src.train --model distilbert --class_weighted --seed $s      # + improvement
done
# optional parameter-efficient comparison (stretch):
# for s in 0 1 2; do python -m src.train --model distilbert --method lora --lr 5e-4 --epochs 6 --seed $s; done
python -m src.evaluate
