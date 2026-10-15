# Assignment 4: random forest for network traffic classification

Compares a tuned random forest with the k-NN and classification tree models
of Assignment 2 on `networkTraffic.csv`.

## Setup

Put `networkTraffic.csv` in this folder (or set `NETWORK_TRAFFIC_CSV` to its
path), then:

    pip install -r requirements.txt

## Run order

    python 01_baselines.py      # Assignment 2 models, 5-fold CV, same seed
    python 02_tuning.py all     # forest tuning, one stage per question
    python 03_final_eval.py     # final comparison on the generalisation set
    python 04_label_overlap.py  # how many observations no model can separate
    python figures.py           # figures from the saved results

Each script reads what the previous one wrote to `results/`, so they must be
run in this order. `02_tuning.py` can also be run one stage at a time, for
example `python 02_tuning.py size_reference`, in the order listed in its
header. Set `N_JOBS` to limit the number of cores, e.g. `N_JOBS=4 python ...`.

## Files

| File | Purpose |
|---|---|
| `common.py` | Seed, data loading, split, folds, preprocessing, measures |
| `01_baselines.py` | Re-evaluates the final k-NN and tree of Assignment 2 |
| `02_tuning.py` | Forest experiments Q1 to Q4 and selection of the final forest |
| `03_final_eval.py` | Cross-validation, generalisation set, out-of-bag estimate, seed repeats, McNemar's test, feature importances |
| `04_label_overlap.py` | Counts observations with identical feature values but different classes |
| `figures.py` | Report figures (PDF and PNG) in `figures/` |

## Design decisions to describe in the report

- **Seed.** `SEED = 42` in `common.py`, the same as Assignment 2. It drives
  the 80:20 split, the five folds and every tree and forest, so the split and
  folds are identical to Assignment 2. `01_baselines.py` should therefore
  reproduce the Assignment 2 cross-validation results.
- **Baselines are not re-tuned.** The k-NN and tree keep the control
  parameters selected in Assignment 2, which were selected on this same
  development set. No model has used the generalisation set for tuning.
- **Same preprocessing as the tree.** The forest uses the tree pipeline of
  Assignment 2 unchanged (no scaling, missing `service` as its own category,
  one-hot encoding, rare `proto` categories grouped).
- **Questions before evaluation.** The four questions and every grid are fixed
  at the top of `02_tuning.py`. Tuning is sequential: each stage starts from
  the configuration selected so far and keeps the setting with the highest
  mean cross-validated macro F1-score.
- **Forest size.** Stages Q2 to Q4 use 100 trees. The final size is the
  smallest forest within 0.001 macro F1 of the best size up to 300 trees.
- **Combination rule.** scikit-learn averages the class-probability estimates
  of the trees. This equals majority voting only when every leaf is pure.
- **Standard deviations** are sample standard deviations over the five folds.

## Repository contents

- `results/` holds the CSV and JSON outputs of the seed-42 run reported in the report.
- `figures/` holds the figures generated from those results.
- `report/` holds the LaTeX source (`report.tex`), the IEEEtran class, the three figures the report includes, and the compiled `report.pdf`.

The dataset `networkTraffic.csv` is not included; place it in this folder before running the scripts.
