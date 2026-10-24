"""Model factories and validation grids for Phase 04 classifiers (RF, LinearSVC, RBF SVM)."""
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src import config

RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20, 40],
    "max_features": ["sqrt", "log2"],
}
LINEAR_SVM_GRID = {"C": [0.01, 0.1, 1]}
RBF_SVM_GRID = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.001],
}


def make_rf(
    n_estimators: int = 100,
    max_depth: int | None = None,
    max_features: str = "sqrt",
) -> Pipeline:
    """Return a new unfitted Random Forest pipeline (no scaler needed for tree models)."""
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=config.SEED,
            n_jobs=config.N_JOBS,
        )),
    ])


def make_linear_svm(C: float = 1.0) -> Pipeline:
    """Return a new unfitted scaled LinearSVC pipeline."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LinearSVC(
            C=C,
            dual="auto",
            random_state=config.SEED,
        )),
    ])


def make_rbf_svm(C: float = 1.0, gamma: Any = "scale") -> Pipeline:
    """Return a new unfitted scaled RBF SVM pipeline."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            random_state=config.SEED,
        )),
    ])
