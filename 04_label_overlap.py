"""How much of the data cannot be separated by any classifier?

Counts the observations whose descriptive feature values are identical to
those of at least one observation from a different class. Such observations
cannot all be classified correctly, whatever the model. The script also
reports the best scores attainable in-sample, by predicting the most frequent
class of every distinct feature vector.

This is a description of the data set (all observations, the 39 descriptive
features that the models use). It is not used for model selection.

Output: results/label_overlap.csv
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from common import CLASS_NAMES, RESULTS_DIR, load_data


def main():
    X, y = load_data()
    frame = X.copy()
    frame["service"] = frame["service"].fillna("Missing")
    key = pd.Series(pd.util.hash_pandas_object(frame, index=False).to_numpy())
    labels = pd.Series(y.to_numpy())

    by_vector = labels.groupby(key)
    in_conflict = by_vector.transform("nunique") > 1
    majority = by_vector.transform(lambda s: s.value_counts().index[0])

    print(f"Observations: {len(labels)}; distinct feature vectors: {key.nunique()}")
    print(f"Observations sharing a feature vector with another class: "
          f"{int(in_conflict.sum())} ({in_conflict.mean():.1%})")
    print(f"Attainable accuracy: {accuracy_score(labels, majority):.4f}; "
          f"attainable macro F1-score: {f1_score(labels, majority, average='macro'):.4f}")

    table = pd.DataFrame({
        "class": CLASS_NAMES,
        "observations": labels.value_counts().sort_index().to_numpy(),
        "share_in_conflict": in_conflict.groupby(labels).mean().to_numpy(),
        "attainable_recall": recall_score(labels, majority, average=None),
        "attainable_f1": f1_score(labels, majority, average=None),
    })
    table.to_csv(RESULTS_DIR / "label_overlap.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
