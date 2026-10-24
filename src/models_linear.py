"""Model factories and validation grids for the Phase 03 classifiers."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


DUMMY_GRID = {}
LOGREG_GRID = {"C": [0.01, 0.1, 1, 10]}
KNN_GRID = {
    "n_neighbors": [1, 3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
}


def make_dummy() -> Pipeline:
    """Return a new unfitted most-frequent-class baseline."""
    return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])


def make_logreg(C: float) -> Pipeline:
    """Return a new unfitted scaled multinomial Logistic Regression model."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=config.LOGREG_MAX_ITER, random_state=config.SEED)),
    ])


def make_knn(n_neighbors: int, weights: str, metric: str) -> Pipeline:
    """Return a new unfitted scaled k-nearest-neighbours model."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=config.N_JOBS,
        )),
    ])


def plot_logreg_weights(C: float):
    """Fit pixel LR on trainval and plot its standardised-pixel class weights."""
    import matplotlib.pyplot as plt
    import numpy as np

    from src.features import get_features, get_labels
    from src.utils import Timer

    model = make_logreg(C)
    with Timer("pixel LR refit for coefficient maps (no test evaluation)"):
        model.fit(get_features("pixels", "trainval"), get_labels("trainval"))
    classifier = model.named_steps["clf"]
    weights = classifier.coef_.reshape(-1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    limit = float(np.abs(weights).max())
    fig, axes = plt.subplots(2, config.NUM_CLASSES // 2, figsize=(12, 5.5), layout="constrained")
    for label, weight, ax in zip(classifier.classes_, weights, axes.flat):
        image = ax.imshow(weight, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_title(config.CLASS_NAMES[label], fontsize=10)
        ax.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Weight per standardised pixel")
    fig.suptitle(f"Pixel Logistic Regression coefficients (C={C})")
    return fig
