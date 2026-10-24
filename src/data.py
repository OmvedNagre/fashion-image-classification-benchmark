"""Getting the Fashion-MNIST data: download, integrity check and loading.

See docs/data-formats.md §2–4 and docs/decisions.md D-003, D-006, D-012.
"""
import gzip
import hashlib
import urllib.request

import numpy as np

from src import config

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
