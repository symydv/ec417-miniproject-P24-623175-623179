# Report outline (4–6 pages, export to report/report.pdf)
1. **Framing (0.5 p)** – task, why macro-F1, class imbalance numbers.
2. **Data & EDA (0.75 p)** – class balance figure, length histogram → max_len 64, duplicate/leakage check, label-noise examples.
3. **Methods (1 p)** – the 4 models, hyper-parameters, class-weight formula w_c = N/(K n_c), diagram, "all models use the same split & metric".
4. **Results (1 p)** – table (mean ± std, 3 seeds), per-class F1 figure, training-time & params, bootstrap CI + McNemar.
5. **Does the improvement help? (0.75 p)** – compare macro-F1 AND surprise/love F1 AND accuracy. If it doesn't help, explain (weights raise recall on rare classes but cut precision; noisy labels; small gain vs seed std). An honest null result gets full credit.
6. **Error analysis (0.75 p)** – confusion matrix, 20 errors categorised (label noise, ambiguity, negation, sarcasm, short text).
7. **Limitations/ethics + what next (0.5 p)** – larger model, label smoothing, ensembling, threshold/prior shift, LoRA.
8. **AI-use statement + references.**
