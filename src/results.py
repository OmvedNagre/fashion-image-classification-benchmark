"""Results aggregation, cross-model comparison plots, and best model selection.

Phase 05 (Results & Analysis) module. Implements:
- load_all_metrics: load metrics JSONs for a split
- pick_best_experiment: select best experiment using comparable validation accuracy
- build_summary: assemble results/summary.csv with exact data-formats.md §11 schema
- plot_pixels_vs_hog: grouped bar comparison of features across models (RQ2)
- plot_acc_vs_inference: accuracy vs inference latency scatter plot with hardware caveats (RQ3)
- plot_recall_heatmap: per-class recall heatmap highlighting hard categories (RQ4)
- Stubs for Omved's Task 05.4 error analysis functions
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, utils


# --- 1. Metrics Loading and Aggregation ---------------------------------------

def load_all_metrics(split: str = "test") -> list[dict]:
    """Load all metrics JSON files for a given split ('val', 'test', or 'cv').

    Reads results/metrics/*_{split}.json.
    Returns list of parsed JSON dicts sorted by experiment filename.
    """
    pattern = f"*_{split}.json"
    files = sorted(config.METRICS_DIR.glob(pattern))
    metrics = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            metrics.append(json.load(f))
    return metrics


def pick_best_experiment() -> str:
    """Pick the best experiment based strictly on validation accuracy (never test accuracy).

    Rule (D-005, D-029):
    For models tuned on a subset (train_subset not null, e.g. RBF SVM), their tuning
    val scores are from a 10k-subset model and not comparable to full 54k models.
    We therefore use the comparable val_acc from results/metrics/{experiment}_val_full.json (D-029).
    For all other models, use best_val_acc from results/metrics/{experiment}_val.json.

    Returns:
        The experiment id string (e.g. 'rbf_svm_hog').
    """
    val_files = sorted(config.METRICS_DIR.glob("*_val.json"))
    best_exp = None
    best_score = -1.0

    for p in val_files:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        exp = d["experiment"]

        # Check for comparable full-training validation score (D-029)
        full_path = config.METRICS_DIR / f"{exp}_val_full.json"
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f_full:
                d_full = json.load(f_full)
            score = float(d_full["val_acc"])
        else:
            score = float(d["best_val_acc"])

        if score > best_score:
            best_score = score
            best_exp = exp

    if best_exp is None:
        raise FileNotFoundError("No validation metric files found in results/metrics/")

    return best_exp


def build_summary() -> pd.DataFrame:
    """Aggregate all 12 test evaluations into results/summary.csv.

    Schema follows docs/data-formats.md §11 exactly:
    experiment, model, features, val_acc, test_acc, test_macro_f1,
    recall_c0, recall_c1, recall_c2, recall_c3, recall_c4,
    recall_c5, recall_c6, recall_c7, recall_c8, recall_c9,
    train_time_s, inference_ms_per_1000, model_size_bytes

    Timing rule (D-024):
    If timing_comparable is False (e.g. logreg experiments where fit overlapped validation),
    retain accuracy/size metrics but emit NaN for train_time_s and inference_ms_per_1000.
    Never substitute zero.

    Returns:
        pd.DataFrame containing the 12 experiment rows.
    """
    test_metrics = load_all_metrics("test")
    if not test_metrics:
        raise FileNotFoundError("No test metric files found in results/metrics/")

    rows = []
    for tm in test_metrics:
        exp = tm["experiment"]
        model = tm["model"]
        features = tm["features"]

        # Comparable validation accuracy (D-029)
        full_path = config.METRICS_DIR / f"{exp}_val_full.json"
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f_full:
                val_data = json.load(f_full)
            val_acc = float(val_data["val_acc"])
        else:
            val_path = config.METRICS_DIR / f"{exp}_val.json"
            with open(val_path, "r", encoding="utf-8") as f_val:
                val_data = json.load(f_val)
            val_acc = float(val_data["best_val_acc"])

        # Predictive metrics from test evaluation
        test_acc = float(tm["accuracy"])
        test_macro_f1 = float(tm["macro_f1"])

        recalls = tm.get("per_class_recall", [np.nan] * 10)

        # Timing comparability check (D-024)
        is_timing_comp = tm.get("timing_comparable", True)
        if is_timing_comp:
            train_time_s = float(tm["train_time_s"])
            inference_ms = float(tm["inference_ms_per_1000"])
        else:
            train_time_s = np.nan
            inference_ms = np.nan

        model_size_bytes = int(tm["model_size_bytes"])

        row = {
            "experiment": exp,
            "model": model,
            "features": features,
            "val_acc": round(val_acc, 4),
            "test_acc": round(test_acc, 4),
            "test_macro_f1": round(test_macro_f1, 4),
            "recall_c0": round(float(recalls[0]), 4),
            "recall_c1": round(float(recalls[1]), 4),
            "recall_c2": round(float(recalls[2]), 4),
            "recall_c3": round(float(recalls[3]), 4),
            "recall_c4": round(float(recalls[4]), 4),
            "recall_c5": round(float(recalls[5]), 4),
            "recall_c6": round(float(recalls[6]), 4),
            "recall_c7": round(float(recalls[7]), 4),
            "recall_c8": round(float(recalls[8]), 4),
            "recall_c9": round(float(recalls[9]), 4),
            "train_time_s": train_time_s if np.isnan(train_time_s) else round(train_time_s, 2),
            "inference_ms_per_1000": inference_ms if np.isnan(inference_ms) else round(inference_ms, 2),
            "model_size_bytes": model_size_bytes,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Order rows in canonical benchmark order
    model_rank = {"dummy": 0, "logreg": 1, "knn": 2, "rf": 3, "linear_svm": 4, "rbf_svm": 5}
    feat_rank = {"pixels": 0, "hog": 1}
    df["_m_rank"] = df["model"].map(lambda m: model_rank.get(m, 99))
    df["_f_rank"] = df["features"].map(lambda f: feat_rank.get(f, 99))
    df = df.sort_values(["_m_rank", "_f_rank"]).drop(columns=["_m_rank", "_f_rank"]).reset_index(drop=True)

    # Write summary table
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.RESULTS_DIR / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"saved {utils.display_path(summary_path)} ({len(df)} experiments)")

    return df


# --- 2. Comparison Visualizations (Task 05.2) ---------------------------------

def plot_pixels_vs_hog(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart comparing Test Accuracy for raw pixels vs HOG (RQ2).

    Saves results/figures/results_pixels_vs_hog.png via utils.save_fig().
    """
    models = ["dummy", "logreg", "knn", "rf", "linear_svm", "rbf_svm"]
    display_names = {
        "dummy": "Dummy",
        "logreg": "Logistic Reg",
        "knn": "k-NN",
        "rf": "Random Forest",
        "linear_svm": "Linear SVM",
        "rbf_svm": "RBF SVM",
    }

    pixel_accs = []
    hog_accs = []
    labels = []

    for m in models:
        row_pix = df[(df["model"] == m) & (df["features"] == "pixels")]
        row_hog = df[(df["model"] == m) & (df["features"] == "hog")]
        if not row_pix.empty and not row_hog.empty:
            labels.append(display_names.get(m, m))
            pixel_accs.append(row_pix["test_acc"].values[0])
            hog_accs.append(row_hog["test_acc"].values[0])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=config.FIGURE_DPI)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

    rects1 = ax.bar(x - width / 2, [a * 100 for a in pixel_accs], width, label="Raw Pixels (784)", color="#4C72B0", edgecolor="none")
    rects2 = ax.bar(x + width / 2, [a * 100 for a in hog_accs], width, label="HOG (1,728)", color="#DD8452", edgecolor="none")

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("RQ2: Effect of Feature Representation on Test Accuracy (Pixels vs HOG)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=11)

    # Annotate bars with accuracy percentage
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    utils.save_fig(fig, "results_pixels_vs_hog")
    return fig


def plot_acc_vs_inference(df: pd.DataFrame) -> plt.Figure:
    """Scatter plot of Test Accuracy vs Inference Latency (RQ3).

    Excludes experiments where timing_comparable=false (D-024) and the dummy
    baseline where zero latency (dummy_pixels = 0.00 ms) cannot be displayed
    on a logarithmic scale (dummy_hog is also excluded to focus on candidate models).
    Preserves original measurements in results/summary.csv.
    Documents hardware differences (M3 vs M3 Pro) and RBF upper bound (D-028).
    Saves results/figures/results_acc_vs_time.png via utils.save_fig().
    """
    # Exclude non-comparable timing (D-024) and dummy baseline with zero/near-zero latency
    valid_df = df[
        df["inference_ms_per_1000"].notna()
        & (df["inference_ms_per_1000"] > 0)
        & (df["model"] != "dummy")
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=config.FIGURE_DPI)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")

    color_map = {"pixels": "#4C72B0", "hog": "#DD8452"}
    marker_map = {"pixels": "o", "hog": "s"}

    label_offsets = {
        "linear_svm_pixels": (10, -8),
        "linear_svm_hog": (10, 4),
        "rf_pixels": (10, -8),
        "rf_hog": (10, 4),
        "knn_hog": (10, -5),
        "knn_pixels": (10, -8),
        "rbf_svm_pixels": (10, 5),
        "rbf_svm_hog": (-115, -12),
    }

    for feat in ["pixels", "hog"]:
        sub = valid_df[valid_df["features"] == feat]
        ax.scatter(
            sub["inference_ms_per_1000"],
            sub["test_acc"] * 100,
            c=color_map[feat],
            marker=marker_map[feat],
            s=90,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.8,
            label=f"Features: {feat.capitalize()}",
            zorder=4,
        )

    for _, row in valid_df.iterrows():
        exp = row["experiment"]
        x_val = row["inference_ms_per_1000"]
        y_val = row["test_acc"] * 100
        offset = label_offsets.get(exp, (10, 0))
        ax.annotate(
            exp,
            xy=(x_val, y_val),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.8, 50000)
    ax.set_ylim(80, 94)
    ax.set_xlabel("Inference Latency (ms per 1,000 images, log scale)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("RQ3: Accuracy vs Inference Speed Trade-off", fontsize=13, pad=12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right", fontsize=10)

    # Footnote detailing exclusions and hardware context
    footnote = (
        "Notes: Excluded from latency plot:\n"
        "• Logistic Regression (pixels, HOG) excluded due to non-comparable timing (D-024).\n"
        "• Dummy baseline (pixels: 0.00 ms, HOG: 0.01 ms) excluded: zero latency cannot be plotted on a log scale;\n"
        "  timings originated from a separate sandbox environment with trivial 10.0% accuracy.\n"
        "Hardware context: Phase 03 timed on Apple M3, Phase 04 on Apple M3 Pro.\n"
        "RBF SVM HOG inference (~28.9s / 1,000) was measured under system contention and is an upper bound (D-028)."
    )
    plt.figtext(0.08, -0.06, footnote, wrap=True, horizontalalignment="left", fontsize=8, color="#444444")

    fig.tight_layout()
    utils.save_fig(fig, "results_acc_vs_time")
    return fig


def plot_recall_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of per-class recall across all 12 benchmark experiments (RQ4).

    Identifies which classes are confused most often (Shirt, T-shirt, Pullover, Coat).
    Saves results/figures/results_recall_heatmap.png via utils.save_fig().
    """
    recall_cols = [f"recall_c{i}" for i in range(10)]
    matrix = df[recall_cols].values

    fig, ax = plt.subplots(figsize=(11, 7), dpi=config.FIGURE_DPI)
    cax = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    # Labels
    ax.set_xticks(range(10))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["experiment"], fontsize=9.5)

    ax.set_title("RQ4: Per-Class Recall Heatmap Across All Experiments", fontsize=13, pad=14)
    ax.set_xlabel("Fashion-MNIST Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Experiment", fontsize=11, labelpad=8)

    # Colorbar
    cbar = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Class Recall", fontsize=10)

    # Annotate each cell with recall value
    for i in range(len(df)):
        for j in range(10):
            val = matrix[i, j]
            # Use white text on dark cells, dark text on light cells
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)

    fig.tight_layout()
    utils.save_fig(fig, "results_recall_heatmap")
    return fig


# --- 3. Error Analysis Placeholders (Omved's Task 05.4) -----------------------

def plot_confusion(experiment: str, normalize: bool = True):
    """Plot confusion matrix for the given experiment (Task 05.4 - Omved)."""
    raise NotImplementedError("plot_confusion is part of Task 05.4 (assigned to Omved).")


def top_confused_pairs(experiment: str, k: int = 5):
    """Return DataFrame of top k confused class pairs (Task 05.4 - Omved)."""
    raise NotImplementedError("top_confused_pairs is part of Task 05.4 (assigned to Omved).")


def plot_misclassified(experiment: str, true_c: int, pred_c: int, n: int = 16):
    """Plot grid of misclassified images for class pair (Task 05.4 - Omved)."""
    raise NotImplementedError("plot_misclassified is part of Task 05.4 (assigned to Omved).")


def model_agreement(exp_a: str, exp_b: str):
    """Compute prediction agreement statistics between two experiments (Task 05.4 - Omved)."""
    raise NotImplementedError("model_agreement is part of Task 05.4 (assigned to Omved).")
