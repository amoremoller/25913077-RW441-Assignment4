import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------
# Global settings
# --------------------------------------------------------------------------

# The same seed as Assignment 2, so the development/generalisation split and
# the five folds are identical to Assignment 2. The Assignment 2 control
# parameters were selected on this development set, and the generalisation
# set has never been used for tuning by any model.
SEED = 42

N_FOLDS = 5
TEST_SIZE = 0.20

# Number of worker threads/processes. Override with e.g. `N_JOBS=8 python ...`.
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("NETWORK_TRAFFIC_CSV", ROOT / "networkTraffic.csv"))
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

TARGET = "attack_cat"

# Removed for every model, as justified in Assignment 2: `id` is a unique
# identifier, `sloss`/`dloss` are near-perfectly correlated with
# `sbytes`/`dbytes`, and `is_ftp_login` holds invalid values and duplicates
# `ct_ftp_cmd`.
DROP_FEATURES = ["id", "is_ftp_login", "sloss", "dloss"]

PROTO_FEATURE = ["proto"]
NOMINAL_FEATURES = ["state", "service"]
BINARY_FEATURES = ["is_sm_ips_ports"]

# Protocol categories below 0.08% of the fitted observations are grouped.
PROTO_MIN_FREQUENCY = 0.0008

CLASS_NAMES = [
    "Normal",
    "Reconnaissance",
    "Backdoor",
    "DoS",
    "Exploits",
    "Analysis",
    "Fuzzers",
    "Worms",
    "Shellcode",
    "Generic",
]

METRICS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]

# Final configurations selected in Assignment 2.
KNN_PARAMS = dict(n_neighbors=7, weights="distance", p=1, algorithm="brute")
TREE_PARAMS = dict(
    criterion="entropy",
    max_depth=30,
    min_samples_leaf=3,
    class_weight=None,
    ccp_alpha=4.641589e-05,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_data():
    """Return (X, y) with the Assignment 2 feature removals applied.

    Missing values are marked by '?' in the file and occur only in `service`.
    """
    df = pd.read_csv(DATA_PATH, na_values="?")
    df = df.drop(columns=DROP_FEATURES)
    for col in PROTO_FEATURE + NOMINAL_FEATURES:
        # Plain object columns with NaN for missing, on every pandas version.
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def split_data(X, y):
    """Stratified 80:20 development/generalisation split."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )


def load_split():
    """Convenience wrapper: X_dev, X_gen, y_dev, y_gen."""
    X, y = load_data()
    return split_data(X, y)


def make_cv():
    """Stratified five-fold cross-validation used for every model."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def numerical_features(X):
    """Every descriptive feature that is not nominal or binary."""
    excluded = set(PROTO_FEATURE + NOMINAL_FEATURES + BINARY_FEATURES + ["service_missing"])
    return [c for c in X.columns if c not in excluded]


# --------------------------------------------------------------------------
# Preprocessing (unchanged from Assignment 2)
# --------------------------------------------------------------------------

def _proto_encoder():
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=PROTO_MIN_FREQUENCY
    )


def make_tree_preprocessor(X):
    """Preprocessing for tree-based models (single tree and random forest).

    Numerical features pass through unscaled. Missing `service` values become
    their own category. Nominal features are one-hot encoded because the
    scikit-learn tree implementation needs numerical input.
    """
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [
            ("num", "passthrough", numerical_features(X)),
            ("proto", _proto_encoder(), PROTO_FEATURE),
            ("cat", nominal, NOMINAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        sparse_threshold=0.0,  # dense output: the matrix is small and mostly numerical
    )


def add_service_missing(X):
    """Add the k-NN missingness indicator for `service` (row-wise, no fitting)."""
    X = X.copy()
    X["service_missing"] = X["service"].isna().astype(int)
    return X


def make_knn_preprocessor(X):
    """Preprocessing for k-NN. Expects X from `add_service_missing`."""
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [
            ("num", MinMaxScaler(), numerical_features(X)),
            ("proto", _proto_encoder(), PROTO_FEATURE),
            ("cat", nominal, NOMINAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES + ["service_missing"]),
        ],
        sparse_threshold=0.0,
    )


# --------------------------------------------------------------------------
# Assignment 2 models
# --------------------------------------------------------------------------

def make_knn_pipeline(X):
    return Pipeline([
        ("preprocessing", make_knn_preprocessor(X)),
        ("knn", KNeighborsClassifier(n_jobs=N_JOBS, **KNN_PARAMS)),
    ])


def make_tree_pipeline(X):
    return Pipeline([
        ("preprocessing", make_tree_preprocessor(X)),
        ("tree", DecisionTreeClassifier(random_state=SEED, **TREE_PARAMS)),
    ])


# --------------------------------------------------------------------------
# Performance measures
# --------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """The five performance measures used in Assignment 2."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_f1(y_true, y_pred):
    """F1-score for each of the ten classes, in class-index order."""
    labels = list(range(len(CLASS_NAMES)))
    return f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)


def per_class_recall(y_true, y_pred):
    labels = list(range(len(CLASS_NAMES)))
    return recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)


def summarise(frame, by):
    """Mean and standard deviation across folds for every numeric column."""
    numeric = [c for c in frame.columns if c not in by and c != "fold"
               and pd.api.types.is_numeric_dtype(frame[c])]
    out = frame.groupby(by, sort=False, dropna=False)[numeric].agg(["mean", "std"])
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    return out.reset_index()
