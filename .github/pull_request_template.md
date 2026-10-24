<!--
Fill EVERY section. Don't delete the checklists; tick them with [x] or explain why a box is unticked.
Title format: "Phase v1/0N — <phase name>". Workflow details: AGENTS.md §6.
-->

## Phase
- **Phase id:** v1/0N — <name>
- **Phase file:** `docs/phases/v1/phase-0N-<slug>.md`
- **Owner:** <your name>
- **AI tool used (if any):** <Claude Code / Antigravity / Cursor / Copilot / none>

## Summary
<!-- 3–6 lines in plain language: what this PR adds, and what someone can now do that they couldn't before. -->

## Tasks completed
<!-- Copy the Tasks list from your phase file, with ticks. One commit per task. -->
- [ ] 0N.1 —
- [ ] 0N.2 —

## Key results
<!-- Numbers and outputs a reviewer should see without running anything. Examples:
     - Phase 01: split sizes, val_indices.npy MD5, duplicate counts, the figures produced
     - Phase 02: chosen HOG params + its val accuracy and dimension
     - Phases 03/04: table of experiment → best params → val acc → test acc → train time → inference ms/1000
     - Phase 05: best model (chosen by val), CV mean ± std, top-3 confused pairs -->
| Item | Value |
|---|---|
| | |

Anything outside the sanity ranges in `docs/prd.md` §8? <!-- yes/no; if yes, explain -->

## No-leakage checklist
- [ ] Test data is only used inside `final_test_run()` (or not touched at all in this phase)
- [ ] Every choice (params / HOG config / best model) was made on **val** scores
- [ ] Scalers are only inside sklearn Pipelines
- [ ] Seed, split file and tuning grids unchanged (or changed with a decisions.md entry)
- [ ] `results/test_log.csv` has one row per experiment (repeats explained below)

## Code and repo checklist
- [ ] Function names/signatures match `docs/module-interfaces.md`, and file formats match `docs/data-formats.md`
- [ ] Notebook ran with **Restart & Run All** with no errors, and is committed with outputs
- [ ] Figures saved via `utils.save_fig` with names following `data-formats.md` §12
- [ ] No new dependencies; no tests/CI/Docker committed; nothing from `.scratch/`, `data/raw/`, `features/` staged
- [ ] Every Done criterion in the phase file is met
- [ ] Every new `decisions.md` entry uses the full template (all headings) and has an Index row
- [ ] Phase file updated: tasks ticked, **Phase summary** filled
- [ ] `docs/progress.md` updated (status `review`, session log entry)

## Decisions logged
<!-- List new entries in docs/decisions.md, e.g. "D-026 — Tuned RBF SVM on a 10k subset because ..." or "none".
     Every entry must use the FULL template (AGENTS.md §6.3): Context, Decision, Alternatives considered, Why,
     How it works, Files touched, Concepts worth learning, Revisit if, plus a row in the Index table. -->
-

## Deviations from the docs
<!-- Anything built differently from the phase file / interfaces / formats, and why. "None" if none. -->
- None

## How to verify
<!-- Exact steps the reviewer can run, usually copied from the phase file's "How to verify". -->
1.

## Notes for the next phase
<!-- What the next phase owner must know: gotchas, slow steps, assumptions. -->
-
