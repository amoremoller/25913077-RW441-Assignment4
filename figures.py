import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLASS_NAMES, FIGURES_DIR, RESULTS_DIR

ENSEMBLE_LABEL = "Random forest"
COLOURS = {"random_forest": "#2a78d6", "classification_tree": "#eb6834", "knn": "#1baf7a"}
MARKERS = {"random_forest": "o", "classification_tree": "s", "knn": "^"}
LABELS = {"random_forest": ENSEMBLE_LABEL, "classification_tree": "Classification tree",
          "knn": "$k$-nearest neighbour"}
NEUTRAL = "#8a8985"
GRID = "#e4e3df"
INK = "#1f1f1e"

COLUMN_WIDTH = 3.5
TEXT_SIZE = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman",
                   "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": TEXT_SIZE,
    "axes.labelsize": TEXT_SIZE,
    "axes.titlesize": TEXT_SIZE,
    "xtick.labelsize": TEXT_SIZE,
    "ytick.labelsize": TEXT_SIZE,
    "legend.fontsize": TEXT_SIZE,
    "axes.edgecolor": NEUTRAL,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.pdf")
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def read(name):
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def legend_above(ax, ncol):
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
              columnspacing=1.0, handlelength=1.4, handletextpad=0.5,
              borderaxespad=0.3)


def figure_ensemble_size():
    reference = read("tuning_size_reference_summary.csv")
    final = read("tuning_size_final_summary.csv")
    baselines = read("baseline_cv_summary.csv")
    if reference is None or final is None or baselines is None:
        return print("skipped ensemble size figure (results missing)")

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.4), layout="constrained")
    series = [
        (reference, "Initial", NEUTRAL, "D"),
        (final, "Selected", COLOURS["random_forest"], MARKERS["random_forest"]),
    ]
    for frame, label, colour, marker in series:
        ax.fill_between(frame["n_trees"],
                        frame["f1_macro_mean"] - frame["f1_macro_std"],
                        frame["f1_macro_mean"] + frame["f1_macro_std"],
                        color=colour, alpha=0.15, linewidth=0)
        ax.plot(frame["n_trees"], frame["f1_macro_mean"], color=colour, marker=marker,
                markersize=3, linewidth=1.2, label=label)

    tree = baselines.set_index("model").loc["classification_tree", "f1_macro_mean"]
    ax.axhline(tree, color=COLOURS["classification_tree"], linewidth=1.2,
               label="Single tree")

    config_path = RESULTS_DIR / "final_config.json"
    if config_path.exists():
        chosen = json.loads(config_path.read_text())["n_estimators"]
        value = final.loc[final["n_trees"] == chosen, "f1_macro_mean"].item()
        ax.scatter([chosen], [value], s=45, facecolor="none",
                   edgecolor=INK, linewidth=0.9, zorder=4, label="Final size")

    ax.set_xscale("log")
    largest = int(final["n_trees"].max())
    ticks = [t for t in [1, 3, 10, 30, 100, 300, 1000] if t <= largest]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Macro F1-score")
    legend_above(ax, ncol=2)
    save(fig, "fig_ensemble_size")


def figure_max_features():
    summary = read("tuning_max_features_summary.csv")
    if summary is None:
        return print("skipped max features figure (results missing)")
    summary = summary.sort_values("features_per_split_mean")
    x = summary["features_per_split_mean"].round().astype(int)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.1), layout="constrained")
    ax.errorbar(x, summary["f1_macro_mean"], yerr=summary["f1_macro_std"],
                color=COLOURS["random_forest"], marker=MARKERS["random_forest"],
                markersize=4, linewidth=1.2, capsize=2, label="Ensemble")
    ax.errorbar(x, summary["tree_f1_macro_mean"], yerr=summary["tree_f1_macro_std"],
                color=COLOURS["classification_tree"],
                marker=MARKERS["classification_tree"], markersize=4, linewidth=1.2,
                capsize=2, label="Mean member tree")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["All" if m == "none" else str(v)
                        for v, m in zip(x, summary["max_features"].astype(str))])
    ax.minorticks_off()
    ax.set_xlabel("Features considered per split")
    ax.set_ylabel("Macro F1-score")
    legend_above(ax, ncol=2)
    save(fig, "fig_max_features")


def figure_class_f1():
    report = read("final_class_report.csv")
    if report is None:
        return print("skipped per-class figure (results missing)")
    models = ["random_forest", "classification_tree", "knn"]
    height = 0.26
    positions = np.arange(len(CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 5.0), layout="constrained")
    for i, model in enumerate(models):
        values = report[report["model"] == model].set_index("class").loc[CLASS_NAMES, "f1"]
        ax.barh(positions + (i - 1) * height, values, height=height * 0.88,
                color=COLOURS[model], label=LABELS[model])
    ax.set_yticks(positions)
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    legend_above(ax, ncol=1)
    save(fig, "fig_class_f1")


def figure_confusion():
    matrix = read("final_confusion_random_forest.csv")
    if matrix is None:
        return print("skipped confusion matrix figure (results missing)")
    counts = matrix.set_index("actual").loc[CLASS_NAMES, CLASS_NAMES].to_numpy()
    share = counts / counts.sum(axis=1, keepdims=True)

    # Every cell shows the percentage of the actual class, so no colour bar
    # is needed.
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.7), layout="constrained")
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if share[i, j] >= 0.005:
                ax.text(j, i, f"{100 * share[i, j]:.0f}", ha="center", va="center",
                        fontsize=TEXT_SIZE,
                        color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "fig_confusion_forest")


def figure_feature_importance(top=10):
    importance = read("final_feature_importance.csv")
    if importance is None:
        return print("skipped feature importance figure (results missing)")
    head = importance.head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.0), layout="constrained")
    ax.barh(head["feature"], head["importance"], height=0.7,
            color=COLOURS["random_forest"])
    ax.set_xlabel("Mean decrease in impurity")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_feature_importance")


if __name__ == "__main__":
    figure_ensemble_size()
    figure_max_features()
    figure_class_f1()
    figure_confusion()
    figure_feature_importance()
