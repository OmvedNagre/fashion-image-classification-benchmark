# Phase v1/03 — Dummy, Logistic Regression & kNN

| | |
|---|---|
| Owner | _TBD_ (set when the phase starts) |
| Status | not-started |
| Branch | `phase/v1-03-linear-knn` |
| Depends on | Phase v1/02 |
| PRD features | F-09 |
| idea.md workstream | WS3 — Linear & Instance-Based Models |

> **New to this repo? Read [`AGENTS.md`](../../../AGENTS.md) first.** It has setup, the no-leakage rules, and the branch → commit → PR workflow. Open the PR using `.github/pull_request_template.md`.

## Goal
By the end of this phase, the dummy baseline, Logistic Regression and kNN are tuned on val and evaluated once on test for
both pixels and HOG (6 test JSONs). Validation curves and LR weight maps explain how these models behave.

## Previous phase summary
<!-- Filled by the owner at phase start (AGENTS.md §6.1): copy the essentials of phase 02's "Phase summary" here. -->

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
- [ ] 03.1 — `src/models_linear.py` with factories + grids → `feat(models): add dummy, logistic regression and kNN factories`
- [ ] 03.2 — Dummy baseline runs (both features) → `feat(linear): run dummy baseline`
- [ ] 03.3 — LR tuning + final test (both features) + validation curves → `feat(linear): tune and evaluate logistic regression`
- [ ] 03.4 — kNN tuning + final test (both features) + validation curves → `feat(linear): tune and evaluate kNN`
- [ ] 03.5 — LR weight heatmaps + notebook write-up (softmax/cross-entropy, role of C, why kNN is slow at predict time, curse of dimensionality) → `feat(linear): add LR weight maps and analysis`
- [ ] 03.6 — Update docs touched by this phase → `docs(phase-03): update docs and decisions`

## Done criteria
- [ ] 6 `*_val.json` + 6 `*_test.json` (dummy, logreg, knn × pixels, hog)
- [ ] `test_log.csv` has exactly one row per experiment
- [ ] Pixel accuracies within the sanity ranges (or a logged explanation)
- [ ] `03_linear_knn.ipynb` runs top-to-bottom
- [ ] Docs match what was built; decisions logged

## How to verify
1. `results/test_log.csv`: 6 rows, no repeated experiment.
2. Scratch check: every `_test.json` has all the schema keys, `len(y_pred) == 10000`, and the confusion matrix sums to 10,000.
3. Notebook 03 → Restart & Run All. Tip: do your test-free tuning runs first; run the `final_test_run` cells only once
   at the end. Every re-run appends to `test_log.csv`, so explain any repeat in `decisions.md` and the PR (never edit the log by hand).

## Phase summary
- **Built:**
- **Key decisions:**
- **Known limitations:**
- **Next phase should know:**

## Next phase preview
Phase 04 adds Random Forest, LinearSVC and RBF SVM through the same harness, with RBF-SVM tuning on a 10k subset.
