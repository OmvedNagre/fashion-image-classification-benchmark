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
