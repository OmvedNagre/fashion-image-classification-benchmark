# Phase v1/04 — Random Forest & SVM

| | |
|---|---|
| Owner | Tanmay Singh |
| Status | in-progress |
| Branch | `phase/v1-04-rf-svm` |
| Depends on | Phase v1/03 |
| PRD features | F-10, S-01 |
| idea.md workstream | WS4 — Tree Ensemble & Kernel Models |

> **New to this repo? Read [`AGENTS.md`](../../../AGENTS.md) first.** It has setup, the no-leakage rules, and the branch → commit → PR workflow. Open the PR using `.github/pull_request_template.md`.

## Goal
By the end of this phase, Random Forest, LinearSVC and RBF SVM are tuned on val and evaluated once on test for both feature
sets. All 8 core experiments (plus the dummy and LinearSVC) are finished.

## Previous phase summary
From v1/03 (PR #1 by Rajat, merged 2026-10-24):
- **Results so far (test):** Dummy 10%; LR pixels 84.72% / HOG 88.86% (C=0.01); kNN pixels 86.11% (k=3, distance, Manhattan) /
  HOG 86.31% (k=3, distance, Euclidean). HOG helps LR a lot (+4.1 pts), kNN barely (+0.2). Shirt recall is lowest everywhere (57–69%).
- **Pattern to copy:** factories in `src/models_tree_kernel.py` return *unfitted* Pipelines + grid constants (see
  `models_linear.py`). Model files must not load data, train or plot (D-025). Plots go in the notebook.
- **Notebook pattern (D-021):** reuse saved `_val.json`/`_test.json` on restart; call `final_test_run` only when the test JSON
  is missing and the experiment isn't already in `test_log.csv`. Copy the guard cells from notebook 03.
- **Finish all tuning for a model before its `final_test_run`** and never run them in parallel (AGENTS.md §4.6, D-023).
- **Timing:** LR timings are flagged `timing_comparable=false` (D-024). Make sure nothing else runs during your final
  fits/predicts so RF/SVM timings stay clean. RBF SVM tunes on `train_subset=config.SVM_TUNE_SUBSET` (10k).
- **Exact LinearSVC is very slow on 54k × 1,728 HOG** (phase 02 note). Budget time, use `dual=False`, or log a documented subset.
- **Log every decision in the full template** (AGENTS.md §6.3). Next free number: **D-026**.

## In scope — exact features
1. **Model factories** — `src/models_tree_kernel.py`
   - Acceptance: `make_rf`, `make_linear_svm`, `make_rbf_svm` as listed in `module-interfaces.md`. Grids:
     `RF_GRID = {n_estimators: [100, 300], max_depth: [None, 20, 40], max_features: [sqrt, log2]}`,
     `LINEAR_SVM_GRID = {C: [0.01, 0.1, 1]}`, `RBF_SVM_GRID = {C: [0.1, 1, 10], gamma: [scale, 0.01, 0.001]}`.
2. **Random Forest** (both features)
   - Acceptance: tuned, refit, tested once. Pixel accuracy is around 87–88%. Pixel feature-importance 28×28 heatmap.
3. **LinearSVC** (both features) (S-01)
   - Acceptance: tuned, refit, tested once.
4. **RBF SVM** (both features)
   - Acceptance: tuned on a stratified 10k train subset (`train_subset=SVM_TUNE_SUBSET`) and scored on the full 6k val.
     Refit on all 60k, tested once. Pixel accuracy is around 89–90%. The support-vector count is in `model_size_detail`. The
     subsample strategy and runtimes are written up in the notebook.

## Out of scope
- 5-fold CV of the best model and cross-model plots (phase 05: "best" must be chosen across *all* models by val).
- PCA speed-up (parked, used only if a run is impractical. Logged if used).

## Relevant docs
- `module-interfaces.md` → `models_tree_kernel.py`, `experiment.py`
- `system-design.md` → Flow C notes (SVM subset), §7 failure modes (memory/time), §11 trade-offs
- `decisions.md` → D-005, D-008

## Tasks
- [x] 04.1 — `src/models_tree_kernel.py` with factories + grids → `feat(models): add random forest and SVM factories`
- [x] 04.2 — RF tuning + final test (both features) + importance heatmap → `feat(treekernel): tune and evaluate random forest`
- [x] 04.3 — LinearSVC tuning + final test (both features) → `feat(treekernel): tune and evaluate linear SVM`
- [ ] 04.4 — RBF SVM subset tuning + final 60k refit + test (both features) → `feat(treekernel): tune and evaluate RBF SVM`
- [ ] 04.5 — Notebook write-up (bagging, Gini, why there's no scaling for RF, margin/hinge loss, kernel trick, C/gamma, SVM scaling cost) → `feat(treekernel): add RF and SVM analysis`
- [ ] 04.6 — Update docs touched by this phase → `docs(phase-04): update docs and decisions`

## Done criteria
- [ ] 6 more `*_val.json` + `*_test.json` pairs (rf, linear_svm, rbf_svm × pixels, hog)
- [ ] `test_log.csv`: one row per experiment, 12 in total
- [ ] Pixel accuracies within the sanity ranges (or a logged explanation)
- [ ] Runtimes of the slow fits recorded in the notebook
- [ ] `04_rf_svm.ipynb` runs top-to-bottom (long cells clearly marked with expected runtime)
- [ ] Docs match what was built; decisions logged

## How to verify
1. Same schema scratch check as phase 03, run over all 12 test JSONs.
2. Check `rbf_svm_*_val.json` has `"train_subset": 10000`.
3. Notebook 04 → Restart & Run All.

## Phase summary
- **Built:**
- **Key decisions:**
- **Known limitations:**
- **Next phase should know:**

## Next phase preview
Phase 05 aggregates every result, picks the best model by val accuracy, checks its stability with 5-fold CV, and runs
the comparison and error analysis that answer RQ1–RQ4.
