# Phase v1/05 — Results, Error Analysis & README

| | |
|---|---|
| Owner | _TBD_ (set when the phase starts) |
| Status | not-started |
| Branch | `phase/v1-05-results-analysis` |
| Depends on | Phase v1/04 |
| PRD features | F-11, F-12, F-13, F-14 |
| idea.md workstream | WS5 — Evaluation, Error Analysis & Integration |

> **New to this repo? Read [`AGENTS.md`](../../../AGENTS.md) first.** It has setup, the no-leakage rules, and the branch → commit → PR workflow. Open the PR using `.github/pull_request_template.md`.

## Goal
By the end of this phase, every experiment sits in one table, every comparison and error-analysis figure is generated
by code, RQ1–RQ4 are answered in writing, and the README lets anyone reproduce the whole benchmark.
This is v1 (capstone Phase 1) code-complete.

## Previous phase summary
<!-- Filled by the owner at phase start (AGENTS.md §6.1): copy the essentials of phase 04's "Phase summary" here. -->

## In scope — exact features
1. **Aggregation** (F-12)
   - Acceptance: `build_summary()` writes `results/summary.csv` with every experiment and the columns in `data-formats.md` §11.
   - Honour `timing_comparable=false`: keep accuracy/model-size results, but emit NaN for both timing columns. Phase 03's two LR results carry this flag because their final runs overlapped validation work (D-024).
2. **Comparison plots** (F-12)
   - Acceptance: `results_pixels_vs_hog.png` (RQ2), `results_acc_vs_time.png` (RQ3), `results_recall_heatmap.png` (RQ4).
   - Exclude missing times from speed plots/rankings and state which experiments were omitted. Do not treat missing timings as zero or discard their accuracy results.
   - `dummy_*` timings come from a different environment (`env.cpu` = `arm64`, likely a sandbox). They're near zero anyway; footnote them.
3. **Stability check** (F-11)
   - Acceptance: `pick_best_experiment()` picks by **val** accuracy. `cross_validate_best()` writes `_cv.json` with mean ± std.
     If it's an SVM, it uses a 10k stratified subset, documented.
4. **Error analysis** (F-13)
   - Acceptance: normalised confusion matrix of the best model, a top-5 confused-pairs table, a misclassified-image grid for
     each pair, and the agreement between the two best models (same mistakes?). A written explanation of why handcrafted
     features fail on shirt / T-shirt / pullover / coat, linked to the phase 01 EDA findings.
5. **Write-up of RQ1–RQ4 + Phase 2 motivation**
   - Limitation to mention: LR's validation accuracy rises monotonically as C shrinks and the best C (0.01) is the
     smallest value in the planned grid, so the true optimum may be lower (D-020). Not re-tuned, to avoid another test run.
   - Acceptance: `05_results_analysis.ipynb` ends with one short answer per RQ and a "why a CNN next" paragraph.
6. **README** (F-14)
   - Acceptance: setup (conda + VS Code, Colab fallback), run order (notebooks 01→05 with rough runtimes), results table
     generated from `summary.csv`, folder map, split MD5, and a note on test-set discipline (`test_log.csv`).

## Out of scope
- IEEE LaTeX report and slides (parked; planned separately).
- Phase 2 CNN work (v2).

## Relevant docs
- `module-interfaces.md` → `results.py`, `experiment.cross_validate_best`
- `data-formats.md` → §8, §9, §11, §12
- `system-design.md` → Flow D, §3 ("best model" picked on val)
- `prd.md` → §3 research questions, §8 success metrics

## Tasks
- [ ] 05.1 — `results.py`: `load_all_metrics`, `build_summary`, `pick_best_experiment` → `feat(results): aggregate metrics into summary table`
- [ ] 05.2 — Comparison plots (RQ2, RQ3, RQ4) → `feat(results): add cross-model comparison plots`
- [ ] 05.3 — 5-fold CV on the best (by val) experiment → `feat(results): add stability check for best model`
- [ ] 05.4 — Error analysis: confusion matrix, top pairs, misclassified grids, model agreement → `feat(errors): add error analysis`
- [ ] 05.5 — `05_results_analysis.ipynb` write-up: RQ answers + Phase 2 motivation → `feat(results): add results and analysis notebook`
- [ ] 05.6 — Full README → `docs(readme): add setup, run order and results`
- [ ] 05.7 — Update docs touched by this phase; progress shows v1 complete → `docs(phase-05): update docs and decisions`

## Done criteria
- [ ] `summary.csv` contains all 12 experiments (dummy, logreg, knn, rf, linear_svm, rbf_svm × pixels, hog)
- [ ] All figures in `results/figures/` are produced by code; none made by hand
- [ ] `test_log.csv` shows each experiment evaluated once (or repeats explained in `decisions.md`)
- [ ] RQ1–RQ4 answered in notebook 05
- [ ] Fresh-clone run following only the README reproduces the README table
- [ ] Docs match what was built; decisions logged

## How to verify
1. Delete `results/summary.csv` and the figures → run notebook 05 → they are regenerated and identical.
2. Scratch check: the best experiment by val is the one used for CV and error analysis, and the test score played no part in choosing it.
3. Read the README top to bottom as a new teammate would. Every command works.

## Phase summary
- **Built:**
- **Key decisions:**
- **Known limitations:**
- **Next phase should know:**

## Next phase preview
v1 is finished. Next: report and slides (parked), then plan capstone Phase 2 as `docs/phases/v2/` (CNN using the same
split and `evaluate()`, ablations, hybrid CNN-features → SVM aimed at the shirt/T-shirt/pullover confusions).
