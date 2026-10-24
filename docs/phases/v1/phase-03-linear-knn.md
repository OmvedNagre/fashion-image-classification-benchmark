# Phase v1/03 — Dummy, Logistic Regression & kNN

| | |
|---|---|
| Owner | Rajat Srivastav |
| Status | review |
| Branch | `phase/v1-03-linear-knn` |
| Depends on | Phase v1/02 |
| PRD features | F-09 |
| idea.md workstream | WS3 — Linear & Instance-Based Models |

> **New to this repo? Read [`AGENTS.md`](../../../AGENTS.md) first.** It has setup, the no-leakage rules, and the branch → commit → PR workflow. Open the PR using `.github/pull_request_template.md`.

## Goal
By the end of this phase, the dummy baseline, Logistic Regression and kNN are tuned on val and evaluated once on test for
both pixels and HOG (6 test JSONs). Validation curves and LR weight maps explain how these models behave.

## Previous phase summary
From v1/02 (merged 2026-10-24):
- **Features:** `features.get_features(kind, split)` with `kind` ∈ {pixels, hog}, `split` ∈ {train, val, trainval}
  (test only via `final_test_run`). Pixels = 784-d in [0,1], unscaled. HOG = **o12_p4_c2, 1,728-d** (D-017, proxy val acc 0.8818).
  Caches live in `features/` (gitignored) and are rebuilt automatically if missing (~15 s for HOG).
- **Harness (`src/experiment.py`):** build each model as a factory returning an *unfitted* Pipeline (put `StandardScaler`
  inside for LR/kNN), then `best, table = tune_on_val(name, make_fn, GRID, kind)` → `final_test_run(name, make_fn, best, kind)`.
  `tune_on_val` writes `{name}_{kind}_val.json` and never reads test. `final_test_run` refits on 60k, calls `evaluate()`,
  writes `{name}_{kind}_test.json` and appends to `results/test_log.csv`, warning on any repeat (D-018).
- **Run `final_test_run` once per experiment, at the end.** `results/metrics/` and `test_log.csv` are empty right now:
  this phase makes the first test-set contact.
- Best = strict max val accuracy. If the top two are within 0.3 points, you may pick the simpler one by hand and log it (D-018).
- `utils.save_fig(fig, "linear_...")` for figures. Copy notebook 01's first cell for imports.
- EDA expects Shirt/Coat/Pullover/T-shirt to be the hard classes; LR pixels sanity range ≈ 84%, kNN ≈ 85–86% (prd.md §8).

## In scope — exact features
1. **Model factories** — `src/models_linear.py`
   - Acceptance: `make_dummy`, `make_logreg`, `make_knn` return new unfitted Pipelines as listed in `module-interfaces.md`.
     Grids: `LOGREG_GRID = {C: [0.01, 0.1, 1, 10]}`,
     `KNN_GRID = {n_neighbors: [1,3,5,7,9], weights: [uniform, distance], metric: [euclidean, manhattan]}`.
2. **Dummy baseline** (both features)
   - Acceptance: `dummy_pixels_test.json` and `dummy_hog_test.json` exist, with accuracy ≈ 10%.
3. **Logistic Regression** (both features)
   - Acceptance: tuned on val, refit on 60k, tested once. Pixel test accuracy is around 84% (otherwise investigate).
     There is a validation curve (val acc vs C) per feature set.
4. **kNN** (both features)
   - Acceptance: tuned on val, refit, tested once. Pixel test accuracy is around 85–86%. There is a validation curve of val acc vs k
     (best weights/metric) per feature set.
5. **LR weight heatmaps** (pixels)
   - Acceptance: `linear_logreg_weights.png`, a 2×5 grid of 28×28 coefficient maps, one per class.

## Out of scope
- L1 vs L2 penalty, kNN on PCA (parked).
- Cross-model comparison plots (phase 05).

## Relevant docs
- `module-interfaces.md` → `models_linear.py`, `experiment.py`
- `data-formats.md` → §7, §8, §10
- `system-design.md` → Flow C, §7 (LR convergence)

## Tasks
- [x] 03.1 — `src/models_linear.py` with factories + grids → `feat(models): add dummy, logistic regression and kNN factories`
- [x] 03.2 — Dummy baseline runs (both features) → `feat(linear): run dummy baseline`
- [x] 03.3 — LR tuning + final test (both features) + validation curves → `feat(linear): tune and evaluate logistic regression`
- [x] 03.4 — kNN tuning + final test (both features) + validation curves → `feat(linear): tune and evaluate kNN`
- [x] 03.5 — LR weight heatmaps + notebook write-up (softmax/cross-entropy, role of C, why kNN is slow at predict time, curse of dimensionality) → `feat(linear): add LR weight maps and analysis`
- [x] 03.6 — Update docs touched by this phase → `docs(phase-03): update docs and decisions`

## Done criteria
- [x] 6 `*_val.json` + 6 `*_test.json` (dummy, logreg, knn × pixels, hog)
- [x] `test_log.csv` has exactly one row per experiment
- [x] Pixel accuracies within the sanity ranges (or a logged explanation)
- [x] `03_linear_knn.ipynb` runs top-to-bottom
- [x] Docs match what was built; decisions logged

## How to verify
1. `results/test_log.csv`: 6 rows, no repeated experiment.
2. Scratch check: every `_test.json` has all the schema keys, `len(y_pred) == 10000`, and the confusion matrix sums to 10,000.
3. Notebook 03 → Restart & Run All. Tip: do your test-free tuning runs first; run the `final_test_run` cells only once
   at the end. Every re-run appends to `test_log.csv`, so explain any repeat in `decisions.md` and the PR (never edit the log by hand).

## Phase summary
- **Built:** Three unfitted Pipeline factories and exact grids; six complete validation/test JSON pairs; four validation curves; 2×5 pixel LR coefficient maps; executed notebook with theory and findings. Scratch checks verified all schemas, grid combinations, validation selections, 10,000 predictions per test, confusion totals, and six unique audit rows. Nine notebook code cells completed without errors; restarting preserved the audit log byte-for-byte. Split MD5 remains `54ff6b78b27109cdfc3ebe7204040a07`.
- **Key results:** Dummy val/test 10% for both features. LR C=0.01: pixels val 86.42%, test 84.72%; HOG val 89.27%, test 88.86%. kNN k=3/distance: pixels Manhattan val 87.47%, test 86.11%; HOG Euclidean val 87.28%, test 86.31%. Pixel LR is near the approximate ~84% target; pixel kNN is 0.11 percentage points above 86%, consistent with the approximate ~85–86% range and validation-selected Manhattan distance. No parameters were changed in response to test scores.
- **Key decisions:** D-020 raises the LR iteration cap to 5,000 after convergence warnings; all rerun grid fits converged. D-021 reuses stored results on notebook restart and stops if metrics are missing for an already logged test. D-022 plots standardised coefficients via a trainval-only refit. D-023 discloses overlap between LR final fits and the tail of the convergence-only validation rerun; selected C values came from complete initial validation sweeps and were confirmed unchanged.
- **Known limitations:** Single holdout validation; LR optimum may lie beyond the smallest prescribed C. kNN prediction is slow (pixels ~6,927 ms/1,000; HOG ~731 ms/1,000) and stores ~189/415 MB respectively. Timings vary with hardware/background load; LR timings include concurrent validation load (D-023). Weight maps are discriminative templates, not causal explanations. Direct upstream push was denied, so the branch is published from Rajat's fork for review.
- **Next phase should know:** Wait for this PR to merge. Preserve the six test results and append-only log; notebook restarts load them instead of retesting. Use the unchanged split/HOG configuration and shared harness. Keep scaling in Pipelines; use the Phase 04 documented SVM tuning subset. No Phase 04 or Phase 05 implementation was added.

## Next phase preview
Phase 04 adds Random Forest, LinearSVC and RBF SVM through the same harness, with RBF-SVM tuning on a 10k subset.
