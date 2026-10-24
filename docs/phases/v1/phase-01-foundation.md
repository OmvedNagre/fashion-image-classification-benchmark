# Phase v1/01 — Foundation & Data

| | |
|---|---|
| Owner | Rajdeep |
| Status | in-progress |
| Branch | `phase/v1-01-foundation` |
| Depends on | — (first phase) |
| PRD features | F-01, F-02, F-03, F-04 |
| idea.md workstream | WS1 — Data Foundation & EDA |

> **New to this repo? Read [`AGENTS.md`](../../../AGENTS.md) first.** It has setup, the no-leakage rules, and the branch → commit → PR workflow. Open the PR using `.github/pull_request_template.md`.

## Goal
By the end of this phase, anyone can clone the repo, set up the environment, and get the verified dataset with the exact same
54k / 6k / 10k split. An EDA notebook shows which classes will be hard to tell apart.

## Previous phase summary
None. This is the first phase. The repo contains only the docs.

## In scope — exact features
1. **Project skeleton + environment**
   - Acceptance: folders from `architecture.md` §4 exist. `requirements.txt` pins numpy, pandas, scikit-learn,
     scikit-image, matplotlib, ipykernel. `.gitignore` covers `data/raw/`, `data/processed/`, `features/`.
     The README has working setup steps: conda env with Python 3.12, pip install, choosing the kernel in VS Code, plus a Colab fallback.
2. **Config + utils**
   - Acceptance: `config.py` and `utils.py` match `module-interfaces.md`. `save_fig` writes a 300 dpi PNG.
3. **Download + checksum + loader** (F-01)
   - Acceptance: `download_raw()` fetches the 4 files over HTTPS and verifies the official MD5s. A corrupted file is detected,
     deleted and reported. `load_raw()` returns the shapes and dtypes listed in `data-formats.md`.
4. **Sanity checks** (F-03)
   - Acceptance: `run_sanity_checks()` reports shapes, dtypes, label range 0–9, 0 NaNs, 6,000/1,000 per class,
     exact-duplicate count inside train, and the train–test duplicate count.
5. **Fixed split** (F-02)
   - Acceptance: `load_splits()` returns float32 arrays in [0,1]. Val has 6,000 images, 600 per class, and no overlap with train.
     `data/splits/val_indices.npy` is committed and its MD5 is written in the README. A second call loads the
     saved file and does not re-split.
6. **EDA notebook** (F-04)
   - Acceptance: `notebooks/01_data_eda.ipynb` runs top-to-bottom ("Restart & Run All") and uses **the train split only**.
     It saves at least 6 figures (`eda_*.png`): sample grid, per-class mean and std images, intensity histogram, PCA 2-D,
     t-SNE on a ~5k subsample, PCA explained-variance curve with the 90%/95% component counts, class-mean similarity
     matrix. It ends with a written paragraph naming the expected hard classes and why.

## Out of scope
- Features, models, evaluate() (phases 02–04).
- Removing duplicates (we only report them, see D-004).

## Relevant docs
- `data-formats.md` → §2 raw data, §3 split, §4 in-memory splits, §12 figures
- `module-interfaces.md` → `config.py`, `utils.py`, `data.py`
- `system-design.md` → Flow A, §3 leakage controls (EDA on train only), §7 failure modes
- `decisions.md` → D-001, D-003, D-006, D-007, D-009, D-012

## Tasks
- [x] 01.1 — Skeleton folders, `.gitignore`, pinned `requirements.txt`, README setup section → `chore(setup): add project skeleton and environment setup`
- [x] 01.2 — `src/config.py` + `src/utils.py` → `feat(core): add config constants and utils`
- [ ] 01.3 — `download_raw()` with MD5 check + `load_raw()` IDX parser → `feat(data): download and verify Fashion-MNIST IDX files`
- [ ] 01.4 — `run_sanity_checks()` → `feat(data): add dataset sanity checks`
- [ ] 01.5 — Stratified split, `load_splits()`, commit `val_indices.npy`, record its MD5 in the README → `feat(data): add fixed stratified train/val split`
- [ ] 01.6 — `01_data_eda.ipynb` + EDA figures + hard-classes paragraph → `feat(eda): add exploratory data analysis notebook`
- [ ] 01.7 — Update docs touched by this phase (decisions, progress) → `docs(phase-01): update docs and decisions`

## Done criteria
- [ ] Fresh clone + README setup + run notebook 01 works with no errors
- [ ] Official MD5s confirmed against the real download (D-006 updated if any value differs)
- [ ] Split: 54,000 / 6,000 / 10,000, 600 per class in val, no train/val overlap, MD5 of `val_indices.npy` in the README
- [ ] At least 6 `eda_*.png` files in `results/figures/`, plus the hard-classes paragraph
- [ ] Docs match what was built; decisions logged

## How to verify
1. `conda activate fmnist && python -c "from src.data import download_raw, load_splits; download_raw(); s = load_splits(); print({k: v.shape for k, v in s.items()})"`
2. Scratch test (`.scratch/test_data.py`, not committed): asserts split sizes, per-class counts, no overlap, value range,
   dtype, and the same MD5 on a second call. Also corrupts a copy of one file and checks that the MD5 check catches it.
3. Open `notebooks/01_data_eda.ipynb` → Restart & Run All → no errors, and the figures appear in `results/figures/`.

## Phase summary
- **Built:**
- **Key decisions:**
- **Known limitations:**
- **Next phase should know:**

## Next phase preview
Phase 02 builds pixel and HOG features from `load_splits()`, picks the HOG settings on val, and builds the shared
`evaluate()` + `tune_on_val()` / `final_test_run()` harness that every model uses.
