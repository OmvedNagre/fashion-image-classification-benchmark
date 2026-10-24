"""Getting the Fashion-MNIST data: download, integrity check and loading.

See docs/data-formats.md §2–4 and docs/decisions.md D-003, D-006, D-012.
"""
import gzip
import hashlib
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split

from src import config
from src.utils import display_path

IMAGE_HEADER_BYTES = 16  # IDX image files start with magic, count, rows, cols (4 bytes each)
LABEL_HEADER_BYTES = 8   # IDX label files start with magic, count


# --- Download + integrity check ------------------------------------------------

def _md5(path) -> str:
    """MD5 hex digest of a file, read in 1 MB chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(name: str, path) -> None:
    """Download one file to a temporary '.part' file, then rename it, so a crash never leaves a half file."""
    url = config.RAW_URL_BASE + name
    tmp_path = path.with_suffix(path.suffix + ".part")
    print(f"downloading {name} ...")
    try:
        urllib.request.urlretrieve(url, tmp_path)
    except OSError as err:  # covers URLError / HTTPError / disk errors
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {err}. Check your internet connection and re-run.") from err
    tmp_path.rename(path)


def download_raw() -> None:
    """Make sure the 4 official files exist in data/raw/ and match Zalando's published MD5 checksums."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, expected_md5 in config.RAW_FILES.items():
        path = config.DATA_RAW / name
        if not path.exists():
            _download(name, path)
        actual_md5 = _md5(path)
        if actual_md5 != expected_md5:
            path.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {name} (expected {expected_md5}, got {actual_md5}). "
                "The file was deleted; re-run download_raw() to fetch it again."
            )
    print(f"all {len(config.RAW_FILES)} raw files present and verified (MD5 OK)")


# --- Loading -----------------------------------------------------------------

def _read_idx(name: str, header_bytes: int) -> np.ndarray:
    """Read a gzipped IDX file into a flat uint8 array, skipping its header."""
    with gzip.open(config.DATA_RAW / name, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=header_bytes)


def load_raw():
    """Return (X_train_full, y_train_full, X_test, y_test) as stored in the official files.

    Images: uint8, shape (N, 28, 28), values 0–255. Labels: int64, shape (N,), values 0–9.
    """
    missing = [name for name in config.RAW_FILES if not (config.DATA_RAW / name).exists()]
    if missing:
        raise RuntimeError(f"Raw files missing: {missing}. Run download_raw() first.")

    side = config.IMAGE_SIZE
    X_train_full = _read_idx("train-images-idx3-ubyte.gz", IMAGE_HEADER_BYTES).reshape(-1, side, side)
    y_train_full = _read_idx("train-labels-idx1-ubyte.gz", LABEL_HEADER_BYTES).astype(np.int64)
    X_test = _read_idx("t10k-images-idx3-ubyte.gz", IMAGE_HEADER_BYTES).reshape(-1, side, side)
    y_test = _read_idx("t10k-labels-idx1-ubyte.gz", LABEL_HEADER_BYTES).astype(np.int64)
    return X_train_full, y_train_full, X_test, y_test


# --- Sanity checks -------------------------------------------------------------

EXPECTED_TRAIN_PER_CLASS = 6000
EXPECTED_TEST_PER_CLASS = 1000


def _image_hashes(images: np.ndarray) -> list:
    """One MD5 per image, so identical images get identical hashes."""
    return [hashlib.md5(img.tobytes()).hexdigest() for img in images]


def _check_invariants(report: dict) -> None:
    """Raise if the data is not what the official dataset should be."""
    problems = []
    if report["train_shape"] != [60000, 28, 28] or report["test_shape"] != [10000, 28, 28]:
        problems.append("unexpected image shapes")
    if report["label_range"] != [0, 9]:
        problems.append("labels outside 0–9")
    if report["nan_count"] != 0:
        problems.append("NaN values present")
    if set(report["train_class_counts"]) != {EXPECTED_TRAIN_PER_CLASS}:
        problems.append("train is not 6,000 per class")
    if set(report["test_class_counts"]) != {EXPECTED_TEST_PER_CLASS}:
        problems.append("test is not 1,000 per class")
    if problems:
        raise RuntimeError(f"Sanity checks failed: {problems}. Delete data/raw/ and re-run download_raw().")


def run_sanity_checks() -> dict:
    """Check the raw data and report duplicates. Raises RuntimeError if a basic property is wrong.

    Duplicates are only counted, not removed, to stay comparable with published results (D-004).
    The test images are only hashed here; no label or choice is derived from them.
    """
    X_train_full, y_train_full, X_test, y_test = load_raw()
    train_hashes = _image_hashes(X_train_full)
    test_hashes = _image_hashes(X_test)
    train_hash_set = set(train_hashes)

    report = {
        "train_shape": list(X_train_full.shape),
        "test_shape": list(X_test.shape),
        "image_dtype": str(X_train_full.dtype),
        "label_dtype": str(y_train_full.dtype),
        "pixel_range": [int(min(X_train_full.min(), X_test.min())), int(max(X_train_full.max(), X_test.max()))],
        "label_range": [int(min(y_train_full.min(), y_test.min())), int(max(y_train_full.max(), y_test.max()))],
        "nan_count": int(np.isnan(X_train_full.astype(np.float32)).sum() + np.isnan(X_test.astype(np.float32)).sum()),
        "train_class_counts": np.bincount(y_train_full, minlength=config.NUM_CLASSES).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=config.NUM_CLASSES).tolist(),
        "duplicates_within_train": len(train_hashes) - len(train_hash_set),
        "duplicates_within_test": len(test_hashes) - len(set(test_hashes)),
        "test_images_also_in_train": sum(h in train_hash_set for h in test_hashes),
    }
    _check_invariants(report)
    for key, value in report.items():
        print(f"{key:28s} {value}")
    return report


# --- Fixed train/val split (D-003) ---------------------------------------------

def _make_val_indices(y_train_full: np.ndarray) -> np.ndarray:
    """Pick VAL_SIZE indices from the 60k train set, stratified by class, with the fixed seed."""
    all_indices = np.arange(len(y_train_full))
    _, val_indices = train_test_split(
        all_indices, test_size=config.VAL_SIZE, stratify=y_train_full, random_state=config.SEED
    )
    return np.sort(val_indices).astype(np.int64)


def _check_val_indices(val_indices: np.ndarray, y_train_full: np.ndarray) -> None:
    """Raise if the saved split is not exactly 6,000 unique, in-range indices with 600 per class."""
    per_class = config.VAL_SIZE // config.NUM_CLASSES
    counts = np.bincount(y_train_full[val_indices], minlength=config.NUM_CLASSES)
    if (
        len(val_indices) != config.VAL_SIZE
        or len(np.unique(val_indices)) != config.VAL_SIZE
        or val_indices.min() < 0
        or val_indices.max() >= len(y_train_full)
        or set(counts.tolist()) != {per_class}
    ):
        raise RuntimeError(
            f"{config.VAL_INDICES_PATH} is not a valid split. Restore it from git: "
            "git checkout -- data/splits/val_indices.npy"
        )


def _load_or_create_val_indices(y_train_full: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Load the committed split file; create it only if it doesn't exist yet (first run of phase 01)."""
    path = config.VAL_INDICES_PATH
    if path.exists():
        val_indices = np.load(path)
    else:
        val_indices = _make_val_indices(y_train_full)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, val_indices)
        print(f"created {display_path(path)}")
    _check_val_indices(val_indices, y_train_full)
    if verbose:
        print(f"val_indices.npy MD5: {_md5(path)}")
    return val_indices


def _to_float(images: np.ndarray) -> np.ndarray:
    """uint8 0–255 -> float32 0–1 (returns a new array)."""
    return images.astype(np.float32) / 255.0


def load_splits(verbose: bool = True) -> dict:
    """Return the fixed splits: train 54k, val 6k (from the official train set) and the official 10k test.

    Keys: X_train, y_train, X_val, y_val, X_test, y_test. Images are float32 in [0, 1], shape (N, 28, 28).
    verbose=False skips printing the split file's MD5 (used by features.py, which calls this often).
    """
    X_train_full, y_train_full, X_test, y_test = load_raw()
    val_indices = _load_or_create_val_indices(y_train_full, verbose)
    is_val = np.zeros(len(y_train_full), dtype=bool)
    is_val[val_indices] = True
    return {
        "X_train": _to_float(X_train_full[~is_val]),
        "y_train": y_train_full[~is_val],
        "X_val": _to_float(X_train_full[is_val]),
        "y_val": y_train_full[is_val],
        "X_test": _to_float(X_test),
        "y_test": y_test,
    }
