"""Random forest tuning experiments.

Each experiment answers one question that was fixed before any forest was
evaluated:

    Q1  How many trees are needed before performance levels off?
    Q2  How does the number of features considered per split affect the
        strength of the individual trees, their diversity, and the ensemble?
    Q3  Do the trees inside the forest need complexity control (leaf size,
        depth, pruning), as the single tree of Assignment 2 did?
    Q4  Does class weighting help the minority attack classes?

Tuning is sequential. Every experiment starts from the configuration selected
so far, varies one group of control parameters, and keeps the setting with the
highest mean cross-validated macro F1-score (the Assignment 2 criterion).

Stages (run in this order, or all at once with `all`):

    size_reference   Q1 on the reference forest (justifies the tuning size)
    max_features     Q2
    complexity       Q3
    class_weight     Q4
    size_final       Q1 on the selected forest (fixes the final size)

Usage:
    python 02_tuning.py all
    python 02_tuning.py max_features complexity

Only the development set is used. Preprocessing is fitted on the training
part of each fold. Outputs go to results/tuning_*.csv and
results/final_config.json.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from common import (
    CLASS_NAMES,
    N_JOBS,
    RESULTS_DIR,
    SEED,
    TREE_PARAMS,
    compute_metrics,
    load_split,
    make_cv,
    make_tree_preprocessor,
    per_class_f1,
    per_class_recall,
    summarise,
)

# --------------------------------------------------------------------------
# Experimental settings (fixed before running)
# --------------------------------------------------------------------------

# Reference forest: the textbook random forest. Bootstrap samples the size of
# the training set, fully grown unpruned trees, sqrt(d) features per split.
# Entropy is kept from the Assignment 2 tree so that the single tree and the
# forest members use the same impurity measure.
REFERENCE = dict(
    criterion="entropy",
    max_features="sqrt",
    min_samples_leaf=1,
    max_depth=None,
    ccp_alpha=0.0,
    class_weight=None,
)

N_TREES_TUNING = 100   # forest size while tuning Q2 to Q4
N_TREES_MAX = 300      # largest forest in the ensemble-size experiments
SIZE_CHECKPOINTS = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300]

# Final size: the smallest forest whose mean macro F1-score is within this
# tolerance of the best mean macro F1-score over all checkpoints.
SIZE_TOLERANCE = 0.001

# Q2: number of features examined at each split. "sqrt" is the reference and
# None means every feature, which turns the forest into plain bagging.
MAX_FEATURES_GRID = [4, "sqrt", 16, 32, 48, None]

# Q3: complexity control, one factor at a time, plus the exact settings that
# were selected for the single tree in Assignment 2.
COMPLEXITY_GRID = [
    ("fully grown", dict(min_samples_leaf=1, max_depth=None, ccp_alpha=0.0)),
    ("leaf >= 3", dict(min_samples_leaf=3, max_depth=None, ccp_alpha=0.0)),
    ("leaf >= 5", dict(min_samples_leaf=5, max_depth=None, ccp_alpha=0.0)),
    ("leaf >= 10", dict(min_samples_leaf=10, max_depth=None, ccp_alpha=0.0)),
    ("leaf >= 25", dict(min_samples_leaf=25, max_depth=None, ccp_alpha=0.0)),
    ("depth <= 10", dict(min_samples_leaf=1, max_depth=10, ccp_alpha=0.0)),
    ("depth <= 20", dict(min_samples_leaf=1, max_depth=20, ccp_alpha=0.0)),
    ("depth <= 30", dict(min_samples_leaf=1, max_depth=30, ccp_alpha=0.0)),
    ("Assignment 2 tree settings", dict(
        min_samples_leaf=TREE_PARAMS["min_samples_leaf"],
        max_depth=TREE_PARAMS["max_depth"],
        ccp_alpha=TREE_PARAMS["ccp_alpha"],
    )),
]

# Q4: class weighting.
CLASS_WEIGHT_GRID = [None, "balanced", "balanced_subsample"]

# Diversity is measured as the mean pairwise disagreement between the
# predictions of the first N_DIVERSITY_TREES trees of each forest.
N_DIVERSITY_TREES = 20

STATE_PATH = RESULTS_DIR / "tuning_state.json"
PARAM_COLUMNS = ["max_features", "min_samples_leaf", "max_depth", "ccp_alpha", "class_weight"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"selected": dict(REFERENCE)}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def prepare_folds():
    """Preprocess every fold once; the fitted statistics come from the
    training part of the fold only."""
    X_dev, _, y_dev, _ = load_split()
    folds = []
    for train_idx, val_idx in make_cv().split(X_dev, y_dev):
        pre = make_tree_preprocessor(X_dev)
        X_train = np.ascontiguousarray(pre.fit_transform(X_dev.iloc[train_idx]), dtype=np.float32)
        X_val = np.ascontiguousarray(pre.transform(X_dev.iloc[val_idx]), dtype=np.float32)
        folds.append((X_train, y_dev.iloc[train_idx].to_numpy(),
                      X_val, y_dev.iloc[val_idx].to_numpy()))
    print(f"Prepared {len(folds)} folds; {folds[0][0].shape[1]} encoded features "
          f"in fold 1.", flush=True)
    return folds


def pairwise_disagreement(tree_predictions):
    """Mean proportion of observations on which two trees disagree."""
    values = []
    for i in range(len(tree_predictions)):
        for j in range(i + 1, len(tree_predictions)):
            values.append(np.mean(tree_predictions[i] != tree_predictions[j]))
    return float(np.mean(values)) if values else np.nan


def evaluate_forest(params, n_trees, fold, checkpoints):
    """Fit one forest on one fold and score it at every checkpoint.

    A forest of the first n trees is itself a forest of n trees, because the
    trees are grown independently. Accumulating the class-probability
    estimates tree by tree therefore gives the validation performance at
    every ensemble size from a single fit. (scikit-learn combines trees by
    averaging class probabilities, which reduces to majority voting when
    every leaf is pure.)
    """
    X_train, y_train, X_val, y_val = fold

    start = time.time()
    forest = RandomForestClassifier(
        n_estimators=n_trees, bootstrap=True, n_jobs=N_JOBS, random_state=SEED, **params
    )
    forest.fit(X_train, y_train)
    fit_seconds = time.time() - start

    classes = forest.classes_
    proba_sum = np.zeros((len(y_val), len(classes)))
    tree_f1, tree_predictions, size_rows = [], [], []

    for i, tree in enumerate(forest.estimators_, start=1):
        proba = tree.predict_proba(X_val)
        proba_sum += proba
        tree_pred = classes[proba.argmax(axis=1)]
        tree_f1.append(f1_score(y_val, tree_pred, average="macro", zero_division=0))
        if i <= N_DIVERSITY_TREES:
            tree_predictions.append(tree_pred)
        if i in checkpoints:
            y_pred = classes[proba_sum.argmax(axis=1)]
            size_rows.append({"n_trees": i, **compute_metrics(y_val, y_pred)})

    y_pred = classes[proba_sum.argmax(axis=1)]
    final = {"n_trees": n_trees, **compute_metrics(y_val, y_pred)}
    for cls, f1, rec in zip(CLASS_NAMES, per_class_f1(y_val, y_pred),
                            per_class_recall(y_val, y_pred)):
        final[f"f1_{cls}"] = f1
        final[f"recall_{cls}"] = rec
    train_scores = compute_metrics(y_train, forest.predict(X_train))
    final.update(
        train_f1_macro=train_scores["f1_macro"],
        f1_gap=train_scores["f1_macro"] - final["f1_macro"],
        tree_f1_macro=float(np.mean(tree_f1)),
        tree_disagreement=pairwise_disagreement(tree_predictions),
        tree_depth=float(np.mean([t.get_depth() for t in forest.estimators_])),
        tree_leaves=float(np.mean([t.get_n_leaves() for t in forest.estimators_])),
        features_per_split=int(forest.estimators_[0].max_features_),
        fit_seconds=fit_seconds,
    )
    return final, size_rows


def param_labels(params):
    """Control parameters as plain values for the CSV files."""
    labels = {k: ("none" if params[k] is None else params[k]) for k in PARAM_COLUMNS}
    for key in ("max_features", "max_depth", "class_weight"):
        labels[key] = str(labels[key])   # these mix numbers and words
    return labels


def run_grid(name, configs, folds, n_trees):
    """Cross-validate a list of (label, params) configurations."""
    rows = []
    for label, params in configs:
        for fold_number, fold in enumerate(folds, start=1):
            final, _ = evaluate_forest(params, n_trees, fold, checkpoints=())
            rows.append({"config": label, "fold": fold_number, **param_labels(params), **final})
            print(f"[{name}] {label:28s} fold {fold_number}: "
                  f"macro F1 = {final['f1_macro']:.4f} ({final['fit_seconds']:.0f} s)",
                  flush=True)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"tuning_{name}_folds.csv", index=False)

    frame = pd.DataFrame(rows)
    summary = summarise(frame, by=["config"] + PARAM_COLUMNS)
    summary.to_csv(RESULTS_DIR / f"tuning_{name}_summary.csv", index=False)
    show = ["config", "f1_macro_mean", "f1_macro_std", "balanced_accuracy_mean",
            "precision_macro_mean", "accuracy_mean", "f1_gap_mean",
            "tree_f1_macro_mean", "tree_disagreement_mean", "tree_leaves_mean"]
    print(f"\n[{name}] summary over {len(folds)} folds")
    print(summary[show].round(4).to_string(index=False), "\n", flush=True)
    return summary


def select_best(summary, configs):
    """Configuration with the highest mean cross-validated macro F1-score."""
    best_label = summary.loc[summary["f1_macro_mean"].idxmax(), "config"]
    return best_label, dict(configs)[best_label]


def run_size(name, params, folds):
    """Validation performance against ensemble size for one configuration."""
    rows = []
    for fold_number, fold in enumerate(folds, start=1):
        final, size_rows = evaluate_forest(params, N_TREES_MAX, fold, SIZE_CHECKPOINTS)
        rows += [{"fold": fold_number, **r} for r in size_rows]
        print(f"[{name}] fold {fold_number}: macro F1 at {N_TREES_MAX} trees = "
              f"{final['f1_macro']:.4f} ({final['fit_seconds']:.0f} s)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / f"tuning_{name}_folds.csv", index=False)
    summary = summarise(frame, by=["n_trees"])
    summary.to_csv(RESULTS_DIR / f"tuning_{name}_summary.csv", index=False)
    print(f"\n[{name}] macro F1-score against number of trees")
    print(summary[["n_trees", "f1_macro_mean", "f1_macro_std",
                   "balanced_accuracy_mean", "accuracy_mean"]].round(4).to_string(index=False),
          "\n", flush=True)
    return summary


def smallest_sufficient_size(summary):
    best = summary["f1_macro_mean"].max()
    enough = summary[summary["f1_macro_mean"] >= best - SIZE_TOLERANCE]
    return int(enough["n_trees"].min())


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_size_reference(folds, state):
    summary = run_size("size_reference", REFERENCE, folds)
    at_tuning = summary.loc[summary["n_trees"] == N_TREES_TUNING, "f1_macro_mean"].item()
    at_max = summary.loc[summary["n_trees"] == N_TREES_MAX, "f1_macro_mean"].item()
    state["size_reference"] = {
        "smallest_sufficient_size": smallest_sufficient_size(summary),
        "f1_at_tuning_size": at_tuning,
        "f1_at_max_size": at_max,
    }
    print(f"Reference forest: macro F1 {at_tuning:.4f} at {N_TREES_TUNING} trees, "
          f"{at_max:.4f} at {N_TREES_MAX} trees.\n", flush=True)


def stage_max_features(folds, state):
    base = state["selected"]
    configs = [(f"max_features={'all' if m is None else m}", {**base, "max_features": m})
               for m in MAX_FEATURES_GRID]
    summary = run_grid("max_features", configs, folds, N_TREES_TUNING)
    label, params = select_best(summary, configs)
    state["selected"] = params
    print(f"Q2 selected: {label}\n", flush=True)


def stage_complexity(folds, state):
    base = state["selected"]
    configs = [(label, {**base, **change}) for label, change in COMPLEXITY_GRID]
    summary = run_grid("complexity", configs, folds, N_TREES_TUNING)
    label, params = select_best(summary, configs)
    state["selected"] = params
    print(f"Q3 selected: {label}\n", flush=True)


def stage_class_weight(folds, state):
    base = state["selected"]
    configs = [(f"class_weight={'none' if w is None else w}", {**base, "class_weight": w})
               for w in CLASS_WEIGHT_GRID]
    summary = run_grid("class_weight", configs, folds, N_TREES_TUNING)
    label, params = select_best(summary, configs)
    state["selected"] = params
    print(f"Q4 selected: {label}\n", flush=True)


def stage_size_final(folds, state):
    summary = run_size("size_final", state["selected"], folds)
    n_trees = smallest_sufficient_size(summary)
    final_config = {**state["selected"], "n_estimators": n_trees}
    state["final_config"] = final_config
    (RESULTS_DIR / "final_config.json").write_text(json.dumps(final_config, indent=2))
    print(f"Final forest: {final_config}\n", flush=True)


STAGES = {
    "size_reference": stage_size_reference,
    "max_features": stage_max_features,
    "complexity": stage_complexity,
    "class_weight": stage_class_weight,
    "size_final": stage_size_final,
}


def main():
    requested = sys.argv[1:] or ["all"]
    names = list(STAGES) if requested == ["all"] else requested
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        sys.exit(f"Unknown stage(s): {unknown}. Choose from {list(STAGES)} or 'all'.")

    print(f"Seed: {SEED}. Stages: {names}", flush=True)
    folds = prepare_folds()
    state = load_state()
    if names[0] == "size_reference":
        state = {"selected": dict(REFERENCE)}   # a full run starts from the reference

    for name in names:
        start = time.time()
        STAGES[name](folds, state)
        save_state(state)
        print(f"Stage '{name}' finished in {(time.time() - start) / 60:.1f} minutes.\n",
              flush=True)


if __name__ == "__main__":
    main()
