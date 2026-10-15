"""Final comparison of the random forest with the Assignment 2 models.

Run after 01_baselines.py and 02_tuning.py. The script

    1. cross-validates the selected forest on the same five folds as the
       baselines (through the complete preprocessing + model pipeline),
    2. fits the forest, the classification tree and the k-NN classifier on
       the whole development set and evaluates each once on the
       generalisation set,
    3. reports the out-of-bag estimate of the forest,
    4. refits the forest with different seeds to measure how much its
       generalisation performance depends on the random number stream,
    5. applies McNemar's test to the paired generalisation predictions,
    6. saves the impurity-based feature importances of the forest.

Outputs go to results/final_*.csv.
"""

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from common import (
    CLASS_NAMES,
    N_JOBS,
    RESULTS_DIR,
    SEED,
    add_service_missing,
    compute_metrics,
    load_split,
    make_cv,
    make_knn_pipeline,
    make_tree_pipeline,
    make_tree_preprocessor,
    per_class_f1,
    summarise,
)

N_SEED_REPEATS = 5   # additional forests grown with seeds SEED+1 ... SEED+5


def load_final_config():
    config = json.loads((RESULTS_DIR / "final_config.json").read_text())
    print("Final forest configuration:", config, flush=True)
    return config


def make_forest(config, seed=SEED, oob_score=False):
    return RandomForestClassifier(
        bootstrap=True, oob_score=oob_score, n_jobs=N_JOBS, random_state=seed, **config
    )


def make_forest_pipeline(X, config, oob_score=False):
    return Pipeline([
        ("preprocessing", make_tree_preprocessor(X)),
        ("forest", make_forest(config, oob_score=oob_score)),
    ])


def cross_validate_forest(config, X_dev, y_dev):
    rows = []
    for fold, (train_idx, val_idx) in enumerate(make_cv().split(X_dev, y_dev), start=1):
        start = time.time()
        model = make_forest_pipeline(X_dev, config)
        model.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        y_val = y_dev.iloc[val_idx]
        y_pred = model.predict(X_dev.iloc[val_idx])

        row = {"model": "random_forest", "fold": fold}
        row.update(compute_metrics(y_val, y_pred))
        for cls, score in zip(CLASS_NAMES, per_class_f1(y_val, y_pred)):
            row[f"f1_{cls}"] = score
        train_scores = compute_metrics(y_dev.iloc[train_idx],
                                       model.predict(X_dev.iloc[train_idx]))
        row.update({f"train_{k}": v for k, v in train_scores.items()})
        trees = model.named_steps["forest"].estimators_
        row.update(depth=np.mean([t.get_depth() for t in trees]),
                   leaves=np.mean([t.get_n_leaves() for t in trees]),
                   nodes=np.mean([t.tree_.node_count for t in trees]))
        row["seconds"] = time.time() - start
        rows.append(row)
        print(f"random_forest        fold {fold}: macro F1 = {row['f1_macro']:.4f} "
              f"({row['seconds']:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def class_report(y_true, y_pred, model):
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame({
        "model": model, "class": CLASS_NAMES, "precision": precision,
        "recall": recall, "f1": f1, "support": support,
    })


def save_confusion(y_true, y_pred, model):
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    frame = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
    frame.index.name = "actual"
    frame.to_csv(RESULTS_DIR / f"final_confusion_{model}.csv")


def mcnemar(y_true, pred_a, pred_b, name_a, name_b):
    """McNemar's test on the observations where exactly one model is correct."""
    correct_a = np.asarray(pred_a) == np.asarray(y_true)
    correct_b = np.asarray(pred_b) == np.asarray(y_true)
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    statistic = (abs(only_a - only_b) - 1) ** 2 / n if n else 0.0
    return {
        "model_a": name_a, "model_b": name_b,
        "only_a_correct": only_a, "only_b_correct": only_b,
        "chi2_continuity": statistic,
        "p_chi2": float(chi2.sf(statistic, df=1)),
        "p_exact": float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0,
    }


def original_feature(name):
    """Map an encoded column name back to the original descriptive feature."""
    group, rest = name.split("__", 1)
    if group == "proto":
        return "proto"
    if group == "cat":
        return "state" if rest.startswith("state") else "service"
    return rest


def main():
    print(f"Seed: {SEED}")
    config = load_final_config()
    X_dev, X_gen, y_dev, y_gen = load_split()

    # ---- 1. Cross-validation on the baseline folds -----------------------
    forest_folds = cross_validate_forest(config, X_dev, y_dev)
    baseline_folds = pd.read_csv(RESULTS_DIR / "baseline_cv_folds.csv")
    all_folds = pd.concat([baseline_folds, forest_folds], ignore_index=True)
    all_folds.to_csv(RESULTS_DIR / "final_cv_folds.csv", index=False)
    cv_summary = summarise(all_folds, by=["model"])
    cv_summary.to_csv(RESULTS_DIR / "final_cv_summary.csv", index=False)
    print("\nCross-validation, same five folds for every model:")
    print(cv_summary[["model", "accuracy_mean", "balanced_accuracy_mean",
                      "precision_macro_mean", "recall_macro_mean",
                      "f1_macro_mean", "f1_macro_std"]].round(4).to_string(index=False))

    # ---- 2. Generalisation set, evaluated once per model -----------------
    predictions = {}
    timings = {}

    start = time.time()
    forest_model = make_forest_pipeline(X_dev, config, oob_score=True).fit(X_dev, y_dev)
    timings["random_forest_fit"] = time.time() - start
    start = time.time()
    predictions["random_forest"] = forest_model.predict(X_gen)
    timings["random_forest_predict"] = time.time() - start

    start = time.time()
    tree_model = make_tree_pipeline(X_dev).fit(X_dev, y_dev)
    timings["classification_tree_fit"] = time.time() - start
    start = time.time()
    predictions["classification_tree"] = tree_model.predict(X_gen)
    timings["classification_tree_predict"] = time.time() - start

    X_dev_knn, X_gen_knn = add_service_missing(X_dev), add_service_missing(X_gen)
    start = time.time()
    knn_model = make_knn_pipeline(X_dev_knn).fit(X_dev_knn, y_dev)
    timings["knn_fit"] = time.time() - start
    start = time.time()
    predictions["knn"] = knn_model.predict(X_gen_knn)
    timings["knn_predict"] = time.time() - start

    order = ["knn", "classification_tree", "random_forest"]
    gen_rows, reports = [], []
    for model in order:
        y_pred = predictions[model]
        gen_rows.append({"model": model, **compute_metrics(y_gen, y_pred),
                         "fit_seconds": timings[f"{model}_fit"],
                         "predict_seconds": timings[f"{model}_predict"]})
        reports.append(class_report(y_gen, y_pred, model))
        save_confusion(y_gen, y_pred, model)
    generalisation = pd.DataFrame(gen_rows)
    generalisation.to_csv(RESULTS_DIR / "final_generalisation.csv", index=False)
    pd.concat(reports, ignore_index=True).to_csv(
        RESULTS_DIR / "final_class_report.csv", index=False)
    pd.DataFrame({"actual": y_gen.to_numpy(), **predictions}).to_csv(
        RESULTS_DIR / "final_generalisation_predictions.csv", index=False)
    print("\nGeneralisation set:")
    print(generalisation.round(4).to_string(index=False))

    # ---- 3. Out-of-bag estimate ------------------------------------------
    forest = forest_model.named_steps["forest"]
    oob = forest.oob_decision_function_
    has_oob = ~np.isnan(oob).any(axis=1)
    oob_pred = forest.classes_[oob[has_oob].argmax(axis=1)]
    oob_scores = compute_metrics(y_dev.to_numpy()[has_oob], oob_pred)
    forest_gen = generalisation.set_index("model").loc["random_forest"]
    forest_cv = cv_summary.set_index("model").loc["random_forest"]
    estimates = pd.DataFrame([
        {"estimate": "out_of_bag", **oob_scores},
        {"estimate": "cross_validation",
         **{m: forest_cv[f"{m}_mean"] for m in oob_scores}},
        {"estimate": "generalisation_set", **{m: forest_gen[m] for m in oob_scores}},
    ])
    estimates.to_csv(RESULTS_DIR / "final_forest_estimates.csv", index=False)
    print("\nForest: three estimates of generalisation performance")
    print(estimates.round(4).to_string(index=False))

    # ---- 4. Dependence on the forest seed --------------------------------
    pre = forest_model.named_steps["preprocessing"]
    X_dev_t = np.ascontiguousarray(pre.transform(X_dev), dtype=np.float32)
    X_gen_t = np.ascontiguousarray(pre.transform(X_gen), dtype=np.float32)
    seed_rows = [{"forest_seed": SEED, **compute_metrics(y_gen, predictions["random_forest"])}]
    for seed in range(SEED + 1, SEED + 1 + N_SEED_REPEATS):
        repeat = make_forest(config, seed=seed).fit(X_dev_t, y_dev)
        seed_rows.append({"forest_seed": seed,
                          **compute_metrics(y_gen, repeat.predict(X_gen_t))})
        print(f"forest seed {seed}: macro F1 = {seed_rows[-1]['f1_macro']:.4f}", flush=True)
        del repeat
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(RESULTS_DIR / "final_forest_seeds.csv", index=False)
    print("\nForest across seeds (generalisation set):")
    print(seeds.drop(columns="forest_seed").agg(["mean", "std", "min", "max"]).round(4))

    # ---- 5. McNemar's test ------------------------------------------------
    tests = pd.DataFrame([
        mcnemar(y_gen, predictions["random_forest"], predictions["classification_tree"],
                "random_forest", "classification_tree"),
        mcnemar(y_gen, predictions["random_forest"], predictions["knn"],
                "random_forest", "knn"),
        mcnemar(y_gen, predictions["classification_tree"], predictions["knn"],
                "classification_tree", "knn"),
    ])
    tests.to_csv(RESULTS_DIR / "final_mcnemar.csv", index=False)
    print("\nMcNemar's test on the generalisation set:")
    print(tests.to_string(index=False))

    # ---- 6. Feature importances -------------------------------------------
    importances = pd.DataFrame({
        "encoded_feature": pre.get_feature_names_out(),
        "importance": forest.feature_importances_,
    })
    importances["feature"] = importances["encoded_feature"].map(original_feature)
    grouped = (importances.groupby("feature")["importance"].sum()
               .sort_values(ascending=False).reset_index())
    grouped.to_csv(RESULTS_DIR / "final_feature_importance.csv", index=False)
    print("\nTen most important descriptive features (impurity-based):")
    print(grouped.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
