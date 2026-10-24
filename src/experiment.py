"""Running experiments: tune on val, then one final test evaluation (F-08).

This is the ONLY module that loads test features, and only inside final_test_run(). tune_on_val() has no way
to receive test data. See docs/system-design.md Flow C and §3, docs/decisions.md D-004, D-005, D-018.
"""
import csv
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score, train_test_split

from src import config, features
from src.evaluate import evaluate
from src.utils import Timer, save_json


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stratified_subset(X: np.ndarray, y: np.ndarray, size: int):
    """A class-balanced random subset of `size` rows (seed 42), kept in original row order."""
    if size >= len(y):
        raise ValueError(f"Subset size {size} must be smaller than the data ({len(y)} rows)")
    idx, _ = train_test_split(np.arange(len(y)), train_size=size, stratify=y, random_state=config.SEED)
    idx = np.sort(idx)
    return X[idx], y[idx]


# --- 1. Tuning on the validation set ----------------------------------------------------

def tune_on_val(name: str, make_fn, grid: dict, kind: str, train_subset: int = None):
    """Fit one model per hyperparameter combo on train (or a stratified subset), score each on val.

    Returns (best_params, table). Best = highest val accuracy; exact ties go to the first combo in grid order.
    Writes results/metrics/{name}_{kind}_val.json. Never loads test data.
    """
    X_train, y_train = features.get_features(kind, "train"), features.get_labels("train")
    X_val, y_val = features.get_features(kind, "val"), features.get_labels("val")
    if train_subset:
        X_train, y_train = stratified_subset(X_train, y_train, train_subset)

    rows = []
    for params in ParameterGrid(grid):
        model = make_fn(**params)
        with Timer(f"{name}_{kind} {params}") as t:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rows.append({"params": params, "fit_s": round(t.seconds, 2),
                     "val_acc": round(float(accuracy_score(y_val, y_pred)), 4),
                     "val_macro_f1": round(float(f1_score(y_val, y_pred, average="macro")), 4)})
        print(f"    val_acc={rows[-1]['val_acc']:.4f}")

    best = max(rows, key=lambda r: r["val_acc"])  # max() keeps the first of equal values
    save_json({"experiment": f"{name}_{kind}", "model": name, "features": kind, "split": "val",
               "train_subset": train_subset, "grid": grid, "rows": rows,
               "best_params": best["params"], "best_val_acc": best["val_acc"], "timestamp": _now()},
              config.METRICS_DIR / f"{name}_{kind}_val.json")
    print(f"best for {name}_{kind}: {best['params']} (val_acc {best['val_acc']:.4f})")
    table = pd.DataFrame([{**r["params"], "val_acc": r["val_acc"], "val_macro_f1": r["val_macro_f1"],
                           "fit_s": r["fit_s"]} for r in rows])
    return dict(best["params"]), table


# --- 2. The single test evaluation ------------------------------------------------------

def _log_test_use(experiment: str, accuracy: float) -> None:
    """Append one row to results/test_log.csv and warn if this experiment was already tested before."""
    path = config.TEST_LOG_PATH
    previous = 0
    if path.exists():
        with open(path, newline="") as f:
            previous = sum(1 for row in csv.DictReader(f) if row["experiment"] == experiment)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "experiment", "accuracy"])
        writer.writerow([_now(), experiment, f"{accuracy:.4f}"])
    if previous:
        print(f"WARNING: {experiment} has now been evaluated on test {previous + 1} times. "
              "Explain the re-run in docs/decisions.md and in the PR (AGENTS.md §4).")


def final_test_run(name: str, make_fn, params: dict, kind: str) -> dict:
    """Refit with the chosen params on all 60k (train + val), evaluate ONCE on test, log it. Returns the metrics."""
    X_trainval, y_trainval = features.get_features(kind, "trainval"), features.get_labels("trainval")
    model = make_fn(**params)
    with Timer(f"final fit {name}_{kind} on {len(y_trainval)} images") as t:
        model.fit(X_trainval, y_trainval)

    X_test, y_test = features.get_features(kind, "test"), features.get_labels("test")  # the only test access
    metrics = evaluate(model, X_test, y_test, name, kind, params, t.seconds)
    _log_test_use(metrics["experiment"], metrics["accuracy"])
    return metrics


# --- 3. Stability check for the best model ------------------------------------------------

def cross_validate_best(name: str, make_fn, params: dict, kind: str, subset: int = None, folds: int = 5) -> dict:
    """Stratified k-fold CV of the whole Pipeline on train + val (or a stratified subset). No test data.

    The scaler is inside the Pipeline, so it is re-fit on each training fold (no leakage between folds).
    Writes results/metrics/{name}_{kind}_cv.json.
    """
    X, y = features.get_features(kind, "trainval"), features.get_labels("trainval")
    if subset:
        X, y = stratified_subset(X, y, subset)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEED)
    with Timer(f"{folds}-fold CV {name}_{kind} on {len(y)} images"):
        scores = cross_val_score(make_fn(**params), X, y, cv=cv, scoring="accuracy", n_jobs=1)
    result = {"experiment": f"{name}_{kind}", "folds": folds, "data": "trainval", "subset": subset,
              "best_params": params, "fold_acc": np.round(scores, 4).tolist(),
              "mean_acc": round(float(scores.mean()), 4), "std_acc": round(float(scores.std()), 4),
              "timestamp": _now()}
    save_json(result, config.METRICS_DIR / f"{name}_{kind}_cv.json")
    print(f"CV accuracy {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
    return result
