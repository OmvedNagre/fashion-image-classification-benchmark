# Decision Log

Every non-trivial decision made while building this project, with the reasoning behind it.
Written so that someone reading it after the project is finished can understand **what** was built,
**why** it was built that way, and **how** the logic works.

Rules:
- Append new entries at the bottom. Never delete an entry; if a decision changes, add a new entry
  and mark the old one `Superseded by D-###`.
- Log as you go, not at the end of a phase.
- Small choices: fill Context, Decision, Why, Files. Architectural or tricky ones: fill everything.

## Index
| ID | Date | Phase | Title | Status |
|---|---|---|---|---|
| D-001 | 2026-10-24 | planning | Python + scikit-learn stack, run locally in VS Code Jupyter | Active |
| D-002 | 2026-10-24 | planning | Logic in small `src/` modules, thin notebooks | Active |
| D-003 | 2026-10-24 | planning | Fixed stratified 54k/6k split with committed val indices | Active |
| D-004 | 2026-10-24 | planning | Leakage controls: Pipelines, test guard, val-based selection | Active |
| D-005 | 2026-10-24 | planning | Tune on a single holdout val set; RBF SVM tuned on a 10k subset | Active |
| D-006 | 2026-10-24 | planning | Verify downloads with Zalando's official MD5 checksums | Active |
| D-007 | 2026-10-24 | planning | Settings as constants in `config.py`; HOG params in cache filenames | Active |
| D-008 | 2026-10-24 | planning | Model size = pickled bytes; inference time per 1,000 test images | Active |
| D-009 | 2026-10-24 | planning | Commit results, figures and notebook outputs; gitignore raw data and caches | Active |
| D-010 | 2026-10-24 | planning | Rename DB/API template docs to data-formats.md and module-interfaces.md | Active |
| D-011 | 2026-10-24 | planning | Store test predictions (y_pred) in each test JSON | Active |
| D-012 | 2026-10-24 | planning | Download raw files over HTTPS from the official GitHub repo | Active |
| D-013 | 2026-10-24 | planning | One owner per phase, delivered as a reviewed PR; AGENTS.md + PR template | Active |

---

## Entry template (copy below the last entry)

### D-### — <Short title>
- **Date:** YYYY-MM-DD
- **Phase:** NN — <phase name>
- **Status:** Active | Superseded by D-### | Reverted
- **Type:** stack | architecture | data | api | algorithm | edge-case | tooling | assumption

**Context**
What problem or question came up, and what constraints mattered.

**Decision**
What was chosen, in one or two sentences.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| | |

**Why**
The reasoning. What made this the right choice for this project at this stage.

**How it works**
Walk through the logic step by step in plain language, as if teaching it. Include a small code
snippet or pseudo-code if it helps. Mention the edge cases it handles.

**Files touched**
- `path/to/file.ts` — what changed

**Concepts worth learning**
Terms, patterns or ideas this decision relies on (e.g. "optimistic concurrency", "idempotency key"),
each with one line explaining it.

**Revisit if**
The conditions under which this decision should be reconsidered.

---

<!-- Entries start below -->

### D-001 — Python + scikit-learn stack, run locally in VS Code Jupyter
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** stack

**Context**
The capstone's Phase 1 must use classical ML only. The team has used Colab and Jupyter before, but hasn't
run notebooks locally. The machine is an M3 MacBook (8 cores, 8 GB RAM) with conda and VS Code installed.

**Decision**
Python 3.12 with NumPy, pandas, scikit-learn, scikit-image and matplotlib. Notebooks run locally in VS Code using a
conda environment. Colab is the documented fallback.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Google Colab as the main platform | Free tier has 2 slow CPUs, sessions time out and wipe state, and importing from `src/` needs a re-clone each session |
| Python 3.14 (already installed) | Newer than Colab (3.12); some wheels may lag. 3.12 keeps local and Colab identical |
| seaborn for plots | One more dependency for little gain; matplotlib does everything we need |
| uv / poetry for envs | Extra tooling to learn; conda is already installed and familiar |

**Why**
scikit-learn is the standard library for classical ML and has every model we need behind one consistent API
(`fit` / `predict`). scikit-image has a reference HOG implementation. Running locally is faster than free Colab
and never loses state.

**How it works**
`conda create -n fmnist python=3.12` → `conda activate fmnist` → `pip install -r requirements.txt` →
open the repo in VS Code → pick the `fmnist` kernel in the notebook. conda provides only the Python
interpreter; all libraries come from pip with pinned versions, so a teammate or Colab gets the same versions.

**Files touched**
- `requirements.txt`, `README.md` (written in phase 01)

**Concepts worth learning**
- Virtual environment: an isolated set of installed packages per project, so versions don't clash.
- Pinning: writing exact versions (`scikit-learn==1.x.y`) so results don't change when a library updates.
- Jupyter kernel: the Python process a notebook runs in; choosing the env's kernel means the notebook uses that env's packages.

**Revisit if**
Phase 2 CNN training is too slow locally. Then use Colab GPU for that phase only.

---

### D-002 — Logic in small `src/` modules, thin notebooks
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** architecture

**Context**
idea.md proposes `src/` files plus one notebook per workstream. We want simple code, no duplicated logic, and
notebooks that are easy to present.

**Decision**
All reusable logic (loading, features, models, tuning, evaluation, aggregation) lives in plain functions in `src/`.
Notebooks import those functions, run them, display plots and hold the written explanations.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Notebooks only | The same split/evaluate code would be copy-pasted into 5 notebooks. One copy drifts, and the results stop being comparable |
| Scripts only | Loses the visual, explainable format the team is used to and the professor expects |
| Config-driven experiment framework (YAML + CLI) | Too much machinery for 9 experiments. Adds learning time with no marks gained |

**Why**
One definition of `load_splits()` and `evaluate()` means every model is guaranteed to use the same data and metrics.
Notebooks stay short and readable, which makes viva preparation easier.

**How it works**
A notebook cell does `from src.experiment import tune_on_val, final_test_run`, then calls them. If the same
code shows up in two notebooks, it gets moved into `src/`. Model files only *build* unfitted sklearn Pipelines
(e.g. `make_knn(k=5, weights="distance")`). They never load data, so they are tiny.

**Files touched**
- `src/*.py`, `notebooks/*.ipynb` (created across phases 01–05)

**Concepts worth learning**
- Separation of concerns: each module has one job (data, features, models, evaluation).
- Factory function: a function that builds and returns a new object (here, an unfitted model) from parameters.

**Revisit if**
A notebook starts holding more than a few lines of non-plotting logic.

---

### D-003 — Fixed stratified 54k/6k split with committed val indices
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** data

**Context**
Hyperparameters must be tuned without touching the official 10k test set, so we need a validation set carved from
the 60k training images. Everyone, on every machine, must get the identical split.

**Decision**
Take 6,000 validation images from the 60k train set with `train_test_split(..., stratify=y, random_state=42)`.
Save the val indices to `data/splits/val_indices.npy` and commit that file (~48 KB). Print an MD5 of it and record that value in the README.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Re-create the split from the seed each time | Usually identical, but a future sklearn/NumPy version could change the shuffle. A committed file can't drift |
| k-fold cross-validation for all tuning | 5× the compute for every model, and SVM would be impractical. See D-005 |
| Non-stratified random split | Class proportions in val could be slightly off. Stratifying guarantees exactly 600 per class |

**Why**
A committed index file is the simplest proof that the split is fixed. The professor can verify that val and test never overlap.

**How it works**
1. `load_splits()` checks whether `val_indices.npy` exists.
2. If not, it builds a stratified split of indices 0–59,999 (seed 42), sorts the val indices and saves them.
3. Train = every index not in val. Test = the official 10k, untouched.
4. A check asserts there are 54,000 train and 6,000 val indices, no overlap, and 600 images per class in val.

**Files touched**
- `src/data.py`, `data/splits/val_indices.npy` (phase 01)

**Concepts worth learning**
- Stratified sampling: sample so that each class keeps its original proportion in every subset.
- Validation vs test set: val is for making choices, test is for reporting the final number once.

**Revisit if**
Never in v1. Phase 2 must reuse this exact file.

---

### D-004 — Leakage controls: Pipelines, test guard, val-based selection
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** algorithm

**Context**
Data leakage (information from val/test sneaking into training or model choice) is the main way an ML project loses
credibility. The user explicitly asked for "no loopholes".

**Decision**
Close every known leak with the simplest mechanism:
1. Put any scaler inside an sklearn `Pipeline` with the model.
2. Only `experiment.final_test_run()` loads test features, and every call is logged to `results/test_log.csv`.
3. Pick the best HOG config, best hyperparameters and "best model" by **val** accuracy only.
4. Run EDA on the train split only.
5. Exact-duplicate images are counted and reported but **not removed**.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Fit `StandardScaler` once on train and save the scaled arrays (idea.md) | Correct for the holdout, but leaks in 5-fold CV (scaler sees all folds) and on the 60k refit unless redone by hand. A Pipeline handles every case automatically |
| Lock the test file behind a password / flag | Over-engineering. A single code path plus an audit log is enough to prove discipline |
| Remove duplicates | Would make our numbers incomparable with published Fashion-MNIST results |

**Why**
Each control is one or two lines of code but closes a specific question an examiner might ask
("was your scaler fit on test?", "how did you pick the best model?").

**How it works**
- `make_logreg(C)` returns `Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C))])`.
  Calling `.fit(X_train)` fits the scaler on X_train only. Calling `.predict(X_val)` only *applies* it.
- `tune_on_val()` takes only train/val arrays. It has no parameter through which test data could arrive.
- `final_test_run()` refits on train+val (60k) with the chosen params, evaluates on test once and appends
  `timestamp, experiment, accuracy` to `test_log.csv`. Duplicate rows for the same experiment would show a re-run.
- HOG has no fitted state (it's a fixed formula applied per image), so computing it on test cannot leak.

**Files touched**
- `src/models_*.py`, `src/experiment.py`, `src/data.py` (phases 01–04)

**Concepts worth learning**
- Data leakage: any path by which information from evaluation data influences training or model choice, inflating scores.
- sklearn Pipeline: chains preprocessing and model so `fit` and `predict` apply each step correctly.
- Stateless vs fitted transform: HOG/flatten have no learned parameters, while a scaler learns mean/std from data.

**Revisit if**
A new preprocessing step that learns from data (e.g. PCA) is added. It must go inside the Pipeline.

---

### D-005 — Tune on a single holdout val set; RBF SVM tuned on a 10k subset
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** algorithm

**Context**
We need to tune about 9 models × several hyperparameters on an 8 GB laptop. RBF-SVM training time grows roughly with the square
of the sample count, so tuning it on 54k samples would take hours.

**Decision**
Tune every model by fitting on the 54k train split and scoring on the 6k val split, looping over a small grid.
The RBF SVM is tuned on a stratified 10k subset of train (scored on the same 6k val). The final models are all refit
on the full 60k. 5-fold stratified CV is run once, on the best config, to report mean ± std.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| `GridSearchCV` with 5-fold CV for every model | 5× compute. SVM/kNN become impractical on a laptop |
| `GridSearchCV` + `PredefinedSplit` | Same result as our loop but harder to read and explain |
| Tune SVM on PCA-reduced pixels | Changes the feature set being compared. Kept as a parked idea |

**Why**
With 6,000 val images, one percentage point of accuracy is 60 images, which is stable enough to rank configs.
A plain `for params in ParameterGrid(grid)` loop is easy to read, prints progress, and gives the data for the
validation curves directly.

**How it works**
```
for params in ParameterGrid(grid):
    model = make_fn(**params); model.fit(X_train, y_train)
    rows.append({**params, "val_acc": accuracy(y_val, model.predict(X_val)), "fit_s": t})
best = row with highest val_acc  (ties → simpler/faster model)
```
For RBF SVM, `X_train` is replaced by a stratified 10k subset drawn with seed 42.

**Files touched**
- `src/experiment.py` (phase 02), used in phases 03–04

**Concepts worth learning**
- Holdout validation: a single fixed slice of data reserved for comparing configurations.
- Validation curve: val score plotted against one hyperparameter, showing under- vs overfitting.
- Refit: after choosing params, retrain on all available training data so the final model sees as much data as possible.

**Revisit if**
Two top configs are within ~0.3% on val. Then confirm with CV before choosing.

---

### D-006 — Verify downloads with Zalando's official MD5 checksums
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** data

**Context**
The PRD draft said SHA-256, but Zalando publishes MD5 checksums for the four files in the official repository README.
We should verify against values published by the source, not values we computed ourselves.

**Decision**
Hard-code the four official MD5 values in `config.py`. `download_raw()` checks each file after download (and on every
run) and stops with an error on a mismatch. PRD F-01 was updated to match.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| SHA-256 computed by us on first download | Only proves the file didn't change after *our* download, not that it was correct |
| No check | A truncated download would give silently wrong data |
| `tensorflow`/`torchvision` dataset loaders | Pulls a huge dependency into a classical-ML phase |

**Why**
MD5 is fine for catching corruption or wrong files. We are not defending against an attacker forging a file,
and these are the checksums the dataset authors publish.

**How it works**
Stream each file through `hashlib.md5()` and compare the hex digest to the expected value. On a mismatch, delete the
file and raise `RuntimeError("checksum mismatch for <file>: re-run to download again")`. The expected values are copied
from github.com/zalandoresearch/fashion-mnist during phase 01 and verified against the actual download.

**Files touched**
- `src/config.py`, `src/data.py` (phase 01); `docs/prd.md` F-01

**Concepts worth learning**
- Checksum / hash: a short fingerprint of a file; any change in the file changes the fingerprint.

**Revisit if**
The official mirror moves. Then update the URL in `config.py`; the checksums stay the same.

---

### D-007 — Settings as constants in `config.py`; HOG params in cache filenames
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** tooling

**Context**
idea.md proposed `config.yaml`. We want one place for the seed, paths and HOG settings, with as few dependencies as possible.

**Decision**
Use a plain Python module `src/config.py` with UPPER_CASE constants. After the HOG sweep in phase 02, the chosen params
are written into `config.HOG_PARAMS` by hand and committed. Feature cache filenames encode the HOG params.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| `config.yaml` + PyYAML | An extra dependency and a parsing step. YAML turns tuples into lists, which needs conversion for `pixels_per_cell` |
| Auto-writing the best HOG config from code | Hides a methodological choice. Picking it by hand from the sweep table is clearer to explain |

**Why**
`from src.config import SEED` is the simplest possible config. Putting params in the cache filename prevents a subtle
bug where old HOG features are reused after params change.

**How it works**
`hog_cache_name(split)` → `f"hog_o{o}_p{p}_c{c}_{split}.npy"` from `HOG_PARAMS`. If the params change, the filename
changes, the cache misses, and the features are rebuilt.

**Files touched**
- `src/config.py` (phase 01), `src/features.py` (phase 02)

**Concepts worth learning**
- Cache invalidation: making sure cached results are thrown away when their inputs change; encoding inputs in the key does this automatically.

**Revisit if**
We need to run many configurations from the command line (not planned).

---

### D-008 — Model size = pickled bytes; inference time per 1,000 test images
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** algorithm

**Context**
RQ3 compares accuracy against speed and size. idea.md measured size differently per model (params, support vectors,
nodes, stored samples), and those numbers can't be put on one axis.

**Decision**
Model size = `len(pickle.dumps(fitted_pipeline))` in bytes, the same for every model. The model-specific count
(support vectors, tree nodes, stored samples, coefficients) is *also* recorded as a secondary field for the write-up.
Inference time = wall-clock time of `predict()` on the full 10k test set, reported as ms per 1,000 images, best of 3 runs.
Train time = wall-clock time of the final 60k fit.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Only model-specific counts | Not comparable across model families |
| `joblib.dump` to disk and measure the file | Same number, but slower and leaves files behind |

**Why**
A serialized size is what a deployment would actually store, so it's fair to compare across models. It also
makes kNN's weakness visible (it stores the whole training set).

**How it works**
`evaluate()` times `model.predict(X_test)` 3 times with `time.perf_counter()` and keeps the minimum (it's the least
affected by background noise). It divides by 10 to get ms per 1,000 images and pickles the model to count its bytes. The
hardware/version info is written into the JSON next to the timings.

**Files touched**
- `src/evaluate.py` (phase 02)

**Concepts worth learning**
- Serialization (pickling): turning an in-memory object into bytes so it can be saved or sent.
- Wall-clock vs CPU time: real elapsed time, which is what a user waiting on predictions experiences.

**Revisit if**
Phase 2 adds GPU models. Then report CPU and GPU timings separately.

---

### D-009 — Commit results, figures and notebook outputs; gitignore raw data and caches
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** tooling

**Context**
The professor will look at the GitHub repo. Raw data and feature caches are large and can be rebuilt; results are small and are the actual deliverable.

**Decision**
Commit `results/` (metrics JSON, `summary.csv`, `test_log.csv`, PNG figures at 300 dpi), `data/splits/val_indices.npy`
and notebooks **with their outputs**. Gitignore `data/raw/`, `data/processed/` and `features/`.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Strip notebook outputs before committing | Cleaner diffs, but the professor would then have to run everything just to see results |
| Commit feature caches | Hundreds of MB. They're rebuilt by code in minutes |

**Why**
Anyone opening the repo on GitHub sees the results immediately, and anyone cloning it can regenerate them.

**How it works**
`.gitignore` lists the three big folders. Each notebook is run top-to-bottom ("Restart & Run All") before
being committed, so its outputs match the current code.

**Files touched**
- `.gitignore` (phase 01)

**Concepts worth learning**
- Build artefacts vs source: generated files that can be rebuilt usually stay out of git; deliverables go in.

**Revisit if**
The repo grows past ~50 MB because of figures.

---

### D-010 — Rename DB/API template docs to data-formats.md and module-interfaces.md
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** tooling

**Context**
The standard docs kit has `database-schema.md` and `api-contracts.md`. This project has no database and no API.
The user pointed out that a developer would open those files expecting a DB and endpoints, and be misled.

**Decision**
Rename them to `data-formats.md` (the files the pipeline saves and what's inside them) and `module-interfaces.md`
(the inputs/outputs of the shared `src/` functions). Update `docs/README.md` and the phase template to match.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Keep template names, explain inside | The filename is the first thing people see; a misleading name costs every reader time |
| Delete both files | We still need one place that pins down the metrics JSON schema and the function signatures shared across phases |

**Why**
File names should describe their content. The content plays the same role as a DB schema and API contract
(a fixed agreement between parts), just for files and functions instead of tables and endpoints.

**Files touched**
- `docs/data-formats.md`, `docs/module-interfaces.md` (renamed), `docs/README.md`, `docs/phases/_phase-template.md`

**Revisit if**
A real database or API is ever added (e.g. a demo app in Phase 2). Then add those docs alongside.

---

### D-011 — Store test predictions (y_pred) in each test JSON
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** data

**Context**
Phase 05 needs misclassified-image grids and a "do the best two models make the same mistakes?" check.
Both need each model's per-image predictions. Retraining models in phase 05 would be slow and would mean
running the test set a second time.

**Decision**
`evaluate()` saves the 10,000 test predictions as `y_pred` in `{name}_{kind}_test.json`.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Save fitted models (`.joblib`) and re-predict | kNN/SVM files are tens of MB each; re-predicting touches test again |
| Separate `.npy` file per model | One more file type to track for ~30 KB of data |

**Why**
Error analysis becomes pure reading of saved results, with no retraining and no extra test-set use.

**How it works**
`y_pred = model.predict(X_test)` is computed once inside `evaluate()` (the same call that's timed), converted to a list of ints
and written with the other metrics. Phase 05 loads it next to `y_test` from `load_splits()`, and `y_pred != y_test`
gives the mistakes.

**Files touched**
- `src/evaluate.py` (phase 02), `src/results.py` (phase 05), `docs/data-formats.md` §8

**Revisit if**
Phase 2 needs prediction probabilities (e.g. for ensembling). Then add `y_proba` the same way.

---

### D-012 — Download raw files over HTTPS from the official GitHub repo
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** data

**Context**
The common Fashion-MNIST mirror is a plain-HTTP S3 website. The same files are also hosted in the official
GitHub repository and served over HTTPS.

**Decision**
`RAW_URL_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"`. Files are verified by MD5 (D-006).

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| S3 HTTP mirror | Unencrypted; works, but HTTPS from the official repo is the better default |

**Why**
Same official files, encrypted transport, and a URL anyone can check in the browser.

**Files touched**
- `src/config.py`, `src/data.py` (phase 01)

**Revisit if**
GitHub rate-limits or blocks the download. Then fall back to the S3 mirror (same MD5s).

---

### D-013 — One owner per phase, delivered as a reviewed PR; AGENTS.md + PR template
- **Date:** 2026-10-24
- **Phase:** planning
- **Status:** Active
- **Type:** tooling

**Context**
The 5 phases will be built by 5 different team members. Some use Claude Code, others use other AI tools
(e.g. Antigravity) or none. The repo owner reviews every phase before it reaches `main`.

**Decision**
Each phase is built on its own branch by its owner, then opened as a GitHub PR into `main` using a fixed PR template,
reviewed by the repo owner (with Claude Code), and merged with a merge commit. `AGENTS.md` holds the tool-neutral rules
and workflow. The project `CLAUDE.md` imports it, so every tool follows the same rules.

**Alternatives considered**
| Option | Why not chosen |
|---|---|
| Rules only in CLAUDE.md | Teammates without Claude would never see them |
| Separate rule files per tool (.cursorrules, etc.) | Copies drift apart; AGENTS.md is a widely read convention and one source |
| Squash-merge PRs | Hides individual commits, but commit history is evidence of each member's contribution |
| CI checks on PRs | Lean profile forbids CI; the PR checklist + owner review replaces it |

**Why**
A phase file says *what* to build. AGENTS.md says *how to work* (setup, leakage rules, branch/commit/PR steps). The PR
template makes every PR summarise its results and tick the leakage checklist, so review is fast and consistent for a human or for Claude.

**How it works**
1. Owner pulls `main`, creates `phase/v1-0N-<slug>`, marks the phase `in-progress` (AGENTS.md §6.1).
2. One commit per task, pushed each time; decisions logged as they happen.
3. Owner closes the phase (done criteria, phase summary, progress → `review`) and opens a PR. GitHub pre-fills the template.
4. Repo owner reviews with Claude (CLAUDE.md lists the review order), the owner fixes with new commits, then it's merged.
5. Next phase starts from the updated `main`.

**Files touched**
- `AGENTS.md`, `.github/pull_request_template.md`, `CLAUDE.md`, `docs/progress.md` (Owner column), all `docs/phases/v1/*.md` (Owner row, AGENTS.md pointer)

**Concepts worth learning**
- Pull request: a request to merge a branch, where code is reviewed and discussed before it reaches `main`.
- Merge commit vs squash: a merge commit keeps every individual commit; squash collapses them into one.

**Revisit if**
Phases need to run in parallel. Then split shared code (harness) earlier, or let later phases branch from an unmerged branch.
