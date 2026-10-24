# Phase v1/02 — Features & Evaluation Harness

| | |
|---|---|
| Owner | Rajdeep |
| Status | in-progress |
| Branch | `phase/v1-02-features-harness` |
| Depends on | Phase v1/01 |
| PRD features | F-05, F-06, F-07, F-08, S-02 |
| idea.md workstream | WS2 — Feature Engineering (+ the WS5 harness, needed before any model runs) |

> **New to this repo? Read [`AGENTS.md`](../../../AGENTS.md) first.** It has setup, the no-leakage rules, and the branch → commit → PR workflow. Open the PR using `.github/pull_request_template.md`.

## Goal
By the end of this phase, every model can get cached pixel or HOG features with one call. The HOG settings are chosen on
val and justified by a sweep table. One shared, leak-proof harness (`tune_on_val` → `final_test_run` → `evaluate`)
exists and is verified, so phases 03 and 04 only need to plug in models.

## Previous phase summary
From v1/01 (merged 2026-10-24):
- `src/data.load_splits()` returns the fixed split: train 54,000 / val 6,000 (600 per class) / test 10,000, as float32 in [0,1],
  shape (N,28,28). Split file `data/splits/val_indices.npy`, MD5 `54ff6b78b27109cdfc3ebe7204040a07`. Never re-split.
- `src/config.py` holds every constant (`SEED`, `N_JOBS`, `SVM_TUNE_SUBSET`, paths). `HOG_PARAMS` is a **placeholder** that this phase replaces.
- `src/utils.py`: `set_seed`, `Timer`, `save_fig(fig, name)` → `results/figures/{name}.png`.
- Data is clean: official MD5s match; 0 duplicates; 0 test images in train.
- EDA (train only): Shirt / Coat / Pullover are nearly identical in pixel space (mean-image cosine ≈ 0.98), so Shirt is the expected
  hardest class. PCA: 84 comps = 90% variance. Worth checking whether HOG separates the tops better.
- Notebooks start with the `sys.path` cell from notebook 01 (AGENTS.md §5). No `data/processed/` cache (D-014).

## In scope — exact features
1. **Pixel + HOG extraction with cache** (F-05)
   - Acceptance: `get_features(kind, split)` works for both kinds and all 4 splits. The second call loads from `features/`.
     Shapes and dtypes match `data-formats.md` §5. HOG params are in the cache filename. No scaling is done here.
2. **HOG parameter sweep** (F-06)
   - Acceptance: `run_hog_sweep()` covers orientations {6,9,12} × ppc {4,7} × cpb {1,2} (12 configs). Each one is scored with a
     proxy linear SVM (SGD, hinge loss, D-016) fit on train and scored on **val**. `results/hog_sweep.csv` is written. The chosen config is put in
     `config.HOG_PARAMS`, and the reasoning is logged in `decisions.md` (accuracy vs dimension vs time).
3. **HOG visualisation** (S-02)
   - Acceptance: `features_hog_examples.png` shows one image per class next to its HOG image.
4. **Evaluation harness** (F-07)
   - Acceptance: `evaluate()` produces every field in `data-formats.md` §8, including `y_pred`, best-of-3 timing,
     `model_size_bytes`, `model_size_detail` and `env`. It writes the JSON.
5. **Experiment runner + test guard** (F-08)
   - Acceptance: `tune_on_val()` loads only train/val and writes `_val.json`. `final_test_run()` refits on `trainval`, calls
     `evaluate()` and appends to `test_log.csv`. `cross_validate_best()` runs stratified K-fold on the whole Pipeline.
     The `train_subset` option draws a stratified subset with seed 42.

## Out of scope
- Any real model experiments, including the dummy (phases 03–04).
- PCA-reduced pixels and LBP (parked).

## Relevant docs
- `data-formats.md` → §5 caches, §6 sweep, §7 val JSON, §8 test JSON, §9 CV JSON, §10 test log
- `module-interfaces.md` → `features.py`, `evaluate.py`, `experiment.py`
- `system-design.md` → Flows B and C, §3 leakage controls
- `decisions.md` → D-004, D-005, D-007, D-008, D-011

## Tasks
- [x] 02.1 — `extract_pixels`, `extract_hog`, `get_features`, `get_labels` with disk cache → `feat(features): add pixel and HOG extractors with caching`
- [x] 02.2 — `run_hog_sweep()`, run it, set `HOG_PARAMS`, log the decision → `feat(features): add HOG parameter sweep and select config`
- [x] 02.3 — `src/evaluate.py` → `feat(eval): add shared evaluation harness`
- [ ] 02.4 — `src/experiment.py` (`tune_on_val`, `final_test_run`, `cross_validate_best`, test log) → `feat(eval): add tuning, final test run and CV helpers`
- [ ] 02.5 — `02_features.ipynb`: sweep table, dimension/time discussion, HOG visualisation, why scaling matters per model → `feat(features): add features notebook`
- [ ] 02.6 — Update docs touched by this phase → `docs(phase-02): update docs and decisions`

## Done criteria
- [ ] Cached features exist for pixels and the chosen HOG config, for all 4 splits
- [ ] `hog_sweep.csv` has 12 rows. The chosen config is justified in `decisions.md`
- [ ] The harness is verified end to end on a tiny model **against a temporary results folder**, so the real
      `results/metrics/` and `test_log.csv` stay empty until phase 03
- [ ] Notebook 02 runs top-to-bottom with no errors
- [ ] Docs match what was built; decisions logged

## How to verify
1. Scratch test (`.scratch/test_harness.py`): point the config paths at a temp folder, run `tune_on_val` + `final_test_run` with
   a DummyClassifier on a small slice, and check that both JSONs match the schema, `test_log.csv` has one row, `y_pred`
   has the right length, and `tune_on_val` never calls `get_features(..., "test")` (wrap it to raise on test).
2. Scratch check: a Pipeline with `StandardScaler` that is fit on train has scaler means equal to the train means, not the val/test means.
3. Notebook 02 → Restart & Run All.

## Phase summary
- **Built:**
- **Key decisions:**
- **Known limitations:**
- **Next phase should know:**

## Next phase preview
Phase 03 writes the model factories for Dummy, Logistic Regression and kNN, then runs them through this harness on both feature sets.
