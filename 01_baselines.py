import time
import pandas as pd

from common import (
    CLASS_NAMES,
    RESULTS_DIR,
    SEED,
    add_service_missing,
    compute_metrics,
    load_split,
    make_cv,
    make_knn_pipeline,
    make_tree_pipeline,
    per_class_f1,
    summarise,
)


def cross_validate(name, make_pipeline, X, y, with_training_scores):
    rows = []
    for fold, (train_idx, val_idx) in enumerate(make_cv().split(X, y), start=1):
        start = time.time()
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = make_pipeline(X)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        row = {"model": name, "fold": fold}
        row.update(compute_metrics(y_val, y_pred))
        for cls, score in zip(CLASS_NAMES, per_class_f1(y_val, y_pred)):
            row[f"f1_{cls}"] = score

        if with_training_scores:
            train_scores = compute_metrics(y_train, model.predict(X_train))
            row.update({f"train_{k}": v for k, v in train_scores.items()})
            tree = model.named_steps["tree"]
            row.update(depth=tree.get_depth(), leaves=tree.get_n_leaves(),
                       nodes=tree.tree_.node_count)

        row["seconds"] = time.time() - start
        rows.append(row)
        print(f"{name:20s} fold {fold}: macro F1 = {row['f1_macro']:.4f} "
              f"({row['seconds']:.0f} s)", flush=True)
    return rows


def main():
    print(f"Seed: {SEED}")
    X_dev, _, y_dev, _ = load_split()
    print("Development set:", X_dev.shape)

    rows = []
    rows += cross_validate("classification_tree", make_tree_pipeline,
                           X_dev, y_dev, with_training_scores=True)
    # Training scores are skipped for k-NN: with distance weighting every
    # training observation is its own nearest neighbour.
    rows += cross_validate("knn", make_knn_pipeline,
                           add_service_missing(X_dev), y_dev,
                           with_training_scores=False)

    folds = pd.DataFrame(rows)
    folds.to_csv(RESULTS_DIR / "baseline_cv_folds.csv", index=False)
    summary = summarise(folds, by=["model"])
    summary.to_csv(RESULTS_DIR / "baseline_cv_summary.csv", index=False)

    print("\nCross-validation summary (mean and standard deviation over folds):")
    cols = ["model", "accuracy_mean", "balanced_accuracy_mean",
            "precision_macro_mean", "recall_macro_mean", "f1_macro_mean", "f1_macro_std"]
    print(summary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
