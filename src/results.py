"""Phase 05: summary table, comparison plots, best-model choice and error analysis.

Reads only saved metrics JSONs (results/metrics/). Never loads test features (AGENTS.md §4).
Tasks 05.1-05.3 (Anand) are implemented; the error-analysis functions (05.4, Omved) are stubs (D-031).
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, utils

MODEL_ORDER = ["dummy", "logreg", "knn", "rf", "linear_svm", "rbf_svm"]
FEATURE_ORDER = ["pixels", "hog"]
RECALL_COLS = [f"recall_c{i}" for i in range(config.NUM_CLASSES)]


# --- 1. Loading, best model and summary table (Task 05.1) ----------------------

def _read_json(path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing {utils.display_path(path)}: run the notebook of the phase that creates it first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_metrics(split: str = "test") -> list[dict]:
    """Read every results/metrics/*_{split}.json ('val', 'test' or 'cv'), sorted by file name."""
    if split not in ("val", "test", "cv"):
        raise ValueError(f"split must be 'val', 'test' or 'cv', got {split!r}")
    return [_read_json(p) for p in sorted(config.METRICS_DIR.glob(f"*_{split}.json"))]


def comparable_val_acc(experiment: str) -> float:
    """Val accuracy from a model trained on all 54k train images (D-029).

    Subset-tuned experiments (RBF SVM) have a {experiment}_val_full.json; every other one uses best_val_acc.
    """
    full_path = config.METRICS_DIR / f"{experiment}_val_full.json"
    if full_path.exists():
        return float(_read_json(full_path)["val_acc"])
    return float(_read_json(config.METRICS_DIR / f"{experiment}_val.json")["best_val_acc"])


def pick_best_experiment() -> str:
    """The experiment with the highest comparable val accuracy. Test scores play no part (D-005, D-029)."""
    experiments = [d["experiment"] for d in load_all_metrics("val")]
    if not experiments:
        raise RuntimeError("No *_val.json files in results/metrics/: run notebooks 03 and 04 first.")
    return max(experiments, key=comparable_val_acc)  # max() keeps the first of equal scores


def build_summary() -> pd.DataFrame:
    """One row per *_test.json with the data-formats.md §11 columns; writes results/summary.csv.

    When timing_comparable is false (the two LR runs, D-024), both timing columns are NaN, never zero.
    """
    test_metrics = load_all_metrics("test")
    if not test_metrics:
        raise RuntimeError("No *_test.json files in results/metrics/: run notebooks 03 and 04 first.")

    rows = []
    for tm in test_metrics:
        timing_ok = tm.get("timing_comparable", True)
        rows.append({
            "experiment": tm["experiment"], "model": tm["model"], "features": tm["features"],
            "val_acc": round(comparable_val_acc(tm["experiment"]), 4),
            "test_acc": round(float(tm["accuracy"]), 4),
            "test_macro_f1": round(float(tm["macro_f1"]), 4),
            **{col: round(float(r), 4) for col, r in zip(RECALL_COLS, tm["per_class_recall"], strict=True)},
            "train_time_s": round(float(tm["train_time_s"]), 2) if timing_ok else np.nan,
            "inference_ms_per_1000": round(float(tm["inference_ms_per_1000"]), 2) if timing_ok else np.nan,
            "model_size_bytes": int(tm["model_size_bytes"]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "features"], key=lambda col: col.map(
        lambda v: (MODEL_ORDER + FEATURE_ORDER).index(v))).reset_index(drop=True)
    path = config.RESULTS_DIR / "summary.csv"
    df.to_csv(path, index=False)
    print(f"saved {utils.display_path(path)} ({len(df)} experiments)")
    return df


# --- 2. Comparison plots (Task 05.2) --------------------------------------------

def plot_pixels_vs_hog(df: pd.DataFrame) -> plt.Figure:
    """RQ2: grouped bars of test accuracy, pixels vs HOG, for every model. Saves results_pixels_vs_hog.png."""
    names = {"dummy": "Dummy", "logreg": "Logistic Reg", "knn": "k-NN", "rf": "Random Forest",
             "linear_svm": "Linear SVM", "rbf_svm": "RBF SVM"}
    acc = df.pivot(index="model", columns="features", values="test_acc").reindex(MODEL_ORDER).dropna() * 100
    x, width = np.arange(len(acc)), 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    for offset, kind, label, color in ((-width / 2, "pixels", "Raw Pixels (784)", "#4C72B0"),
                                       (width / 2, "hog", "HOG (1,728)", "#DD8452")):
        bars = ax.bar(x + offset, acc[kind], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8.5)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("RQ2: Effect of Feature Representation on Test Accuracy (Pixels vs HOG)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in acc.index], fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=11)
    fig.tight_layout()
    utils.save_fig(fig, "results_pixels_vs_hog")
    return fig


def plot_acc_vs_inference(df: pd.DataFrame) -> plt.Figure:
    """RQ3: test accuracy vs inference time per 1,000 images (log x-axis). Saves results_acc_vs_time.png.

    Leaves out experiments with NaN timings (LR, D-024) and the dummy baseline (0.00 ms can't sit on a log
    axis); the footnote names them. Raw timings stay in summary.csv.
    """
    shown = df[df["inference_ms_per_1000"].notna() & (df["model"] != "dummy")]
    # Label positions in points, hand-placed so the labels don't overlap: (dx, dy, horizontal alignment)
    label_pos = {"rbf_svm_hog": (-9, 4, "right"), "rf_pixels": (-9, 5, "right"), "rf_hog": (9, -10, "left")}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    for kind, label, color, marker in (("pixels", "Pixels", "#4C72B0", "o"), ("hog", "HOG", "#DD8452", "s")):
        sub = shown[shown["features"] == kind]
        ax.scatter(sub["inference_ms_per_1000"], sub["test_acc"] * 100, c=color, marker=marker, s=90,
                   edgecolors="black", linewidth=0.8, label=f"{label} features", zorder=4)
    for _, row in shown.iterrows():
        dx, dy, ha = label_pos.get(row["experiment"], (9, -4, "left"))
        ax.annotate(row["experiment"], (row["inference_ms_per_1000"], row["test_acc"] * 100),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlim(0.8, 50000)
    ax.set_ylim(80, 94)
    ax.set_xlabel("Inference time (ms per 1,000 images, log scale)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("RQ3: Accuracy vs Inference Speed Trade-off", fontsize=13, pad=12)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right", fontsize=10)
    fig.tight_layout()

    excluded = ", ".join(df.loc[~df.index.isin(shown.index), "experiment"])
    fig.text(0.02, -0.01, (
        f"Not shown: {excluded}. LR timings are not comparable (D-024); dummy timings are ~0 ms and come from another "
        "machine.\nTimings: phase 03 models (kNN) on an Apple M3, phase 04 models (RF, SVMs) on an Apple M3 Pro, so "
        "cross-phase speed gaps are approximate.\nrbf_svm_hog's 28.9 s / 1,000 looks ~2x slower than expected; "
        "treat it as an upper bound (D-028)."), ha="left", va="top", fontsize=8, color="#444444")
    utils.save_fig(fig, "results_acc_vs_time")
    return fig


def plot_recall_heatmap(df: pd.DataFrame) -> plt.Figure:
    """RQ4: per-class recall of every experiment, one cell per (experiment, class). Saves results_recall_heatmap.png."""
    matrix = df[RECALL_COLS].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["experiment"], fontsize=9.5)
    ax.set_title("RQ4: Per-Class Recall Heatmap Across All Experiments", fontsize=13, pad=14)
    ax.set_xlabel("Fashion-MNIST Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Experiment", fontsize=11, labelpad=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04).set_label("Class Recall", fontsize=10)
    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if matrix[i, j] > 0.65 else "black")  # white text on dark cells
    fig.tight_layout()
    utils.save_fig(fig, "results_recall_heatmap")
    return fig


# --- 3. Error analysis (Task 05.4, Omved, D-031) -------------------------------

def _get_confusion_matrix(experiment: str) -> np.ndarray:
    """Load the stored 10x10 confusion matrix from results/metrics/{experiment}_test.json."""
    data = _read_json(config.METRICS_DIR / f"{experiment}_test.json")
    return np.asarray(data["confusion_matrix"], dtype=np.int64)


def plot_confusion(experiment: str, normalize: bool = True) -> plt.Figure:
    """Normalised or raw confusion matrix of one experiment. Saves errors_confusion_matrix.png.

    Diagonal cells represent correct classifications (recall for that class when normalized).
    Off-diagonal cells represent systematic confusions (misclassifications) between pairs.
    """
    cm = _get_confusion_matrix(experiment)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        matrix = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = ".2f"
        vmax = 1.0
        title_suffix = " (Row-Normalized)"
    else:
        matrix = cm
        fmt = "d"
        vmax = int(cm.max())
        title_suffix = " (Raw Counts)"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_title(f"Confusion Matrix: {experiment}{title_suffix}", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Class", fontsize=11, labelpad=8)
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=9.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion" if normalize else "Count", fontsize=10)

    thresh = vmax / 2.0
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            val = matrix[i, j]
            text = f"{val:{fmt}}" if normalize else f"{int(val)}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=color)

    fig.tight_layout()
    fig_name = "errors_confusion_matrix" if experiment == "rbf_svm_hog" else f"errors_confusion_{experiment}"
    utils.save_fig(fig, fig_name)
    return fig


def top_confused_pairs(experiment: str, k: int = 5) -> pd.DataFrame:
    """Top-k off-diagonal (true, pred) pairs as a DataFrame with columns: (true, pred, count, rate).

    Extracted from the stored test confusion matrix without touching test features.
    """
    cm = _get_confusion_matrix(experiment)
    rows = []
    for i in range(config.NUM_CLASSES):
        row_sum = int(cm[i].sum())
        for j in range(config.NUM_CLASSES):
            if i != j:
                count = int(cm[i, j])
                rate = round(count / row_sum, 4) if row_sum > 0 else 0.0
                rows.append({
                    "true": config.CLASS_NAMES[i],
                    "pred": config.CLASS_NAMES[j],
                    "count": count,
                    "rate": rate,
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["count", "rate"], ascending=[False, False]).head(k).reset_index(drop=True)
    return df


def plot_misclassified(experiment: str, true_c, pred_c, n: int = 16) -> plt.Figure:
    """Grid of test images of class true_c predicted as pred_c (Task 05.4 exception, D-011).

    Uses stored y_pred from results/metrics/{experiment}_test.json and X_test from load_splits().
    true_c and pred_c can be class index (int) or class name (str).
    """
    if isinstance(true_c, str):
        if true_c not in config.CLASS_NAMES:
            raise ValueError(f"Unknown class name: {true_c!r}")
        true_c = config.CLASS_NAMES.index(true_c)
    if isinstance(pred_c, str):
        if pred_c not in config.CLASS_NAMES:
            raise ValueError(f"Unknown class name: {pred_c!r}")
        pred_c = config.CLASS_NAMES.index(pred_c)

    true_name = config.CLASS_NAMES[true_c]
    pred_name = config.CLASS_NAMES[pred_c]

    test_json = _read_json(config.METRICS_DIR / f"{experiment}_test.json")
    y_pred = np.asarray(test_json["y_pred"], dtype=np.int64)

    from src.data import load_splits
    splits = load_splits(verbose=False)
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    matching_idx = np.where((y_test == true_c) & (y_pred == pred_c))[0]
    if len(matching_idx) == 0:
        raise ValueError(f"No test images found where true={true_name} and pred={pred_name}")

    selected_idx = matching_idx[:n]
    ncols = min(4, len(selected_idx))
    nrows = int(np.ceil(len(selected_idx) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = np.asarray(axes).reshape(-1)

    for i, idx in enumerate(selected_idx):
        ax = axes[i]
        ax.imshow(X_test[idx], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Test #{idx}", fontsize=9)
        ax.axis("off")

    for j in range(len(selected_idx), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Misclassified: True '{true_name}' -> Pred '{pred_name}'\n({experiment}, showing {len(selected_idx)} of {len(matching_idx)})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    slug_true = true_name.lower().replace("/", "_").replace(" ", "_")
    slug_pred = pred_name.lower().replace("/", "_").replace(" ", "_")
    fig_name = f"errors_misclassified_{slug_true}_vs_{slug_pred}"
    utils.save_fig(fig, fig_name)
    return fig


def model_agreement(exp_a: str, exp_b: str) -> dict:
    """Agreement counts between two experiments' saved y_pred.

    Returns dict with keys: both_correct, both_incorrect, only_a_incorrect,
    only_b_incorrect, same_wrong_label, total, agreement_pct.
    """
    json_a = _read_json(config.METRICS_DIR / f"{exp_a}_test.json")
    json_b = _read_json(config.METRICS_DIR / f"{exp_b}_test.json")

    pred_a = np.asarray(json_a["y_pred"], dtype=np.int64)
    pred_b = np.asarray(json_b["y_pred"], dtype=np.int64)

    from src.data import load_splits
    splits = load_splits(verbose=False)
    y_test = splits["y_test"]

    corr_a = (pred_a == y_test)
    corr_b = (pred_b == y_test)

    both_correct = int(np.sum(corr_a & corr_b))
    both_incorrect = int(np.sum((~corr_a) & (~corr_b)))
    only_a_incorrect = int(np.sum((~corr_a) & corr_b))
    only_b_incorrect = int(np.sum(corr_a & (~corr_b)))
    same_wrong_label = int(np.sum((~corr_a) & (~corr_b) & (pred_a == pred_b)))
    total = len(y_test)
    agreement_pct = round(float((pred_a == pred_b).mean() * 100), 2)

    return {
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "only_a_incorrect": only_a_incorrect,
        "only_b_incorrect": only_b_incorrect,
        "same_wrong_label": same_wrong_label,
        "total": total,
        "agreement_pct": agreement_pct,
    }
