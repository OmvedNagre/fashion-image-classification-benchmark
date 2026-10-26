# Fashion-MNIST Classical ML Benchmark

A rigorous, leak-free benchmark comparing classical machine learning algorithms on the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) clothing classification dataset. We evaluate four core model families across two distinct feature representations (raw pixels vs. Histogram of Oriented Gradients), evaluate performance along accuracy, latency, and memory dimensions, and conduct comprehensive error analysis to motivate a Phase 2 Convolutional Neural Network (CNN).

> **Contributors:** Built collaboratively across five sequential phases.
> - **Phase 01:** Foundation & Data (Rajdeep)
> - **Phase 02:** Features & Evaluation Harness (Rajdeep)
> - **Phase 03:** Dummy, Logistic Regression & k-NN (Rajat Srivastav)
> - **Phase 04:** Random Forest & Support Vector Machines (Tanmay Singh)
> - **Phase 05:** Results Aggregation, Plots & CV (Anand Mishra, Part A) | Error Analysis, Synthesis & README (Omved Nagre, Part B)
>
> Developer workflow: [`CONTRIBUTING.md`](CONTRIBUTING.md) · Architecture rules & leakage controls: [`AGENTS.md`](AGENTS.md) · Design specs: [`docs/`](docs/)

---

## 1. Dataset & Fixed Data Splits

Fashion-MNIST consists of 70,000 28×28 grayscale images spanning 10 balanced clothing categories (6,000 train / 1,000 test per class):
*0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot.*

### Data Splits
To ensure complete reproducibility and prevent data leakage, the dataset is partitioned as follows:
- **Training split (54,000 images):** Used for model fitting during hyperparameter tuning.
- **Validation split (6,000 images):** Stratified holdout (600 per class, random seed 42) carved from the official 60k training set. Committed to version control at [`data/splits/val_indices.npy`](data/splits/val_indices.npy) to ensure every team member and experiment uses an identical validation partition.
- **Test split (10,000 images):** The official benchmark test set. **Evaluated strictly once per model** via `src.experiment.final_test_run()` after hyperparameter selection is complete.

Verify split checksum:
```bash
md5 -q data/splits/val_indices.npy        # macOS  (Linux/Colab: md5sum data/splits/val_indices.npy)
# expected: 54ff6b78b27109cdfc3ebe7204040a07
```

---

## 2. Environment Setup

### Option A: Local Installation (Recommended)
Requires Python 3.12, conda (or virtualenv), and VS Code.
```bash
git clone https://github.com/rajdeep-2004/fashion-image-classification-benchmark.git
cd fashion-image-classification-benchmark

# Create and activate conda environment
conda create -n fmnist python=3.12 -y
conda activate fmnist

# Install pinned dependencies
pip install -r requirements.txt
```

*Without conda:*
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In VS Code:
1. Press `Cmd/Ctrl+Shift+P` → **Python: Select Interpreter** → select `fmnist` (or `.venv`).
2. When opening any notebook in `notebooks/`, click **Select Kernel** (top right) and pick the `fmnist` interpreter.

### Option B: Google Colab (Fallback)
Run this in the first cell of a Colab notebook:
```python
!git clone https://github.com/rajdeep-2004/fashion-image-classification-benchmark.git
%cd fashion-image-classification-benchmark
!pip install -q -r requirements.txt
```

### Download and Verify Raw Dataset
Download the four official IDX files and automatically verify their published Zalando MD5 checksums:
```bash
python -c "from src.data import download_raw; download_raw()"
```
Raw files are saved in `data/raw/` (gitignored). If any checksum fails, the bad file is purged immediately.

---

## 3. Benchmark Execution Order

Run the notebooks sequentially from `notebooks/`. Each phase notebook relies on artifacts, caches, and models established by preceding phases:

| Step | Notebook | Description | Expected Runtime |
|---|---|---|---|
| 1 | `notebooks/01_data_eda.ipynb` | Data integrity checks, class distributions, PCA projections, and t-SNE manifold analysis on train split | ⏱ ~2–3 min |
| 2 | `notebooks/02_features.ipynb` | Flattened pixels (784-d) extraction, HOG grid sweep across 12 configurations using linear proxy, feature caching | ⏱ ~3–5 min |
| 3 | `notebooks/03_linear_knn.ipynb` | Tuning and evaluation of Dummy baseline, Logistic Regression (L2 penalty), and k-NN (uniform & distance weighting) | ⏱ ~10–15 min |
| 4 | `notebooks/04_rf_svm.ipynb` | Tuning and evaluation of Random Forest (300 trees), LinearSVC, and RBF SVM (10k subset tuning, 60k refit) | ⏱ ~15–20 min |
| 5 | `notebooks/05_results_analysis.ipynb` | Aggregation into `summary.csv`, cross-model comparison plots, 5-fold CV stability, confusion matrix, error analysis, and synthesis | ⏱ <15 s |

*Hardware Note:* Phase 03 was executed on an Apple M3 (8 GB RAM); Phase 04 was executed on an Apple M3 Pro (18 GB RAM). Runtimes will vary with available CPU cores.

---

## 4. Benchmark Results

Aggregated from all 12 completed experiments stored in `results/summary.csv`:

| Experiment | Model | Features | Val Acc | Test Acc | Macro F1 | Train Time (s) | Inference (ms/1k) | Model Size |
|---|---|---|---|---|---|---|---|---|
| `dummy_pixels` | Dummy | Pixels | 0.1000 | 0.1000 | 0.0182 | 0.00 | 0.00* | 709 B |
| `dummy_hog` | Dummy | HOG | 0.1000 | 0.1000 | 0.0182 | 0.16 | 0.01* | 709 B |
| `logreg_pixels` | Logistic Regression | Pixels | 0.8642 | 0.8472 | 0.8464 | NaN† | NaN† | 51.4 KB |
| `logreg_hog` | Logistic Regression | HOG | 0.8927 | 0.8886 | 0.8885 | NaN† | NaN† | 111.8 KB |
| `knn_pixels` | k-NN | Pixels | 0.8747 | 0.8611 | 0.8614 | 0.49 | 6,926.61 | 188.7 MB |
| `knn_hog` | k-NN | HOG | 0.8728 | 0.8631 | 0.8623 | 1.29 | 731.13 | 415.2 MB |
| `rf_pixels` | Random Forest | Pixels | 0.8867 | 0.8756 | 0.8741 | 32.83 | 12.16 | 366.5 MB |
| `rf_hog` | Random Forest | HOG | 0.8735 | 0.8703 | 0.8692 | 68.53 | 19.02 | 500.7 MB |
| `linear_svm_pixels` | Linear SVM | Pixels | 0.8583 | 0.8378 | 0.8353 | 99.24 | 1.22 | 82.7 KB |
| `linear_svm_hog` | Linear SVM | HOG | 0.8900 | 0.8898 | 0.8892 | 162.04 | 2.62 | 180.8 KB |
| `rbf_svm_pixels` | RBF SVM | Pixels | 0.9082 | 0.8970 | 0.8967 | 195.67 | 4,766.94 | 125.1 MB |
| `rbf_svm_hog` | RBF SVM | HOG | **0.9157** | **0.9108** | **0.9108** | 451.43 | 28,876.96‡ | 366.2 MB |

*Footnotes on benchmark measurements:*
- † **Logistic Regression timings:** Preserved as `NaN` (per D-024) because their test runs overlapped validation jobs causing CPU contention. Predictive metrics (accuracy, F1, recall) are fully valid and unaffected.
- \* **Dummy baseline timings:** Recorded in a lightweight sandbox environment (`env.cpu = arm64`).
- ‡ **RBF SVM HOG inference latency:** 28.9 s per 1,000 images is an upper bound (D-028) due to evaluating kernel dot products across >20,000 support vectors in 1,728 dimensions.

---

## 5. Key Findings & Synthesis

### Best Model Selection
The overall best model is **`rbf_svm_hog`**, selected **strictly using comparable validation accuracy** (0.9157 on 54k train, D-029). Test evaluation confirmed this choice with **91.08% test accuracy** and **91.08% Macro F1**. Stratified 5-fold cross-validation on `trainval` yielded a stable **0.8775 ± 0.0054** (the CV score is lower because each fold trains on 8,000 images rather than the full 54,000).

### Core Conclusions
1. **RQ1 (Raw Pixels Ceiling):** Classical models on raw pixels max out at **89.70%** (RBF SVM). Linear models struggle (~83.8%–84.7%) because raw pixel intensity distributions lack geometric structure and translation invariance.
2. **RQ2 (Impact of HOG):** HOG dramatically boosts linear models (**+5.20% for LinearSVC**, **+4.14% for Logistic Regression**) and pushes RBF SVM past 91%. In contrast, Random Forest slightly degrades (**-0.53%**) because axis-aligned decision trees struggle with dense, correlated orientation histograms compared to sparse raw pixel values.
3. **RQ3 (Production Pareto Frontier):**
   - **Low Latency Champion:** `rf_pixels` classifies 1,000 items in **12.16 ms** with 87.56% accuracy and requires no preprocessing scaling.
   - **Compact Linear Compromise:** `linear_svm_hog` achieves **88.98%** accuracy in **2.62 ms** inference time with a compact **180 KB** footprint.
   - **Accuracy Maximum:** `rbf_svm_hog` delivers **91.08%**, but requires **28.9 seconds per 1,000 images**, making it impractical for high-throughput serving without kernel approximations.
4. **RQ4 (Systematic Error Modes):**
   - Errors are heavily concentrated in the upper-body clothing cluster: **Shirt (Class 6), T-shirt/top (Class 0), Pullover (Class 2), and Coat (Class 4)**.
   - **Shirt is universally the hardest class** (lowest recall across all 12 models: 50.9% to 75.2%).
   - The top two confused directed pairs are **Shirt → T-shirt/top** (10.4% error rate) and **T-shirt/top → Shirt** (9.1% error rate). At 28×28 grayscale resolution, collar folds, button plackets, and sleeve cuffs blur into nearly identical silhouettes that fixed-grid HOG extractors cannot disambiguate.

---

## 6. Scientific Rigor & No-Leakage Policy

This benchmark strictly adheres to university capstone scientific controls (`AGENTS.md` §4):
- **Zero Test Set Tuning:** Hyperparameters and feature configurations are selected exclusively on the holdout validation set.
- **Pipeline-Encapsulated Scaling:** `StandardScaler` lives strictly inside scikit-learn `Pipeline` objects. Feature scalers never see validation or test distributions.
- **Append-Only Test Audit Trail:** Every execution of `final_test_run()` appends to `results/test_log.csv`. The log confirms exactly 12 runs (one per experiment) with zero re-tuning leakage.
- **Reproducible Figures:** All figures in `results/figures/` are generated directly from code and tracked in version control.

---

## 7. Future Work: Capstone Phase 2 (CNN)

Classical machine learning has established an empirical ceiling of ~91.1% on Fashion-MNIST. The primary bottleneck is the representational limit of fixed, handcrafted spatial extractors (HOG) on overlapping garment silhouettes (Shirt vs. T-shirt vs. Coat).

**Capstone Phase 2** will introduce deep learning architectures using the identical data split and `evaluate()` harness:
1. **Hierarchical Representation Learning:** Convolutional layers will learn localized feature detectors directly from data, automatically progressing from low-level edges to mid-level collar/button textures and high-level garment compositions.
2. **Translation Equivariance:** Weight sharing in convolutional filters will detect fine garment attributes regardless of slight posture or centering shifts.
3. **Targeted Error Resolution:** A compact 2–3 layer CNN with Batch Normalization and Dropout will be trained to resolve the ~500 mutual errors identified in our error analysis, targeting >93–94% accuracy with sub-millisecond inference.

---

## 8. Repository Layout

```
fashion-image-classification-benchmark/
├── src/                    # Reusable pipeline modules
│   ├── config.py           # Constants, seeds, paths, class names, HOG params
│   ├── utils.py            # Timer, set_seed, save_fig, save_json
│   ├── data.py             # Download, MD5 verification, load_splits
│   ├── features.py         # Pixel flattening, HOG extraction & sweep
│   ├── models_linear.py    # Dummy, Logistic Regression, k-NN Pipelines
│   ├── models_tree_kernel.py # Random Forest, LinearSVC, RBF SVM Pipelines
│   ├── evaluate.py         # Evaluation metrics, latency timing, model sizing
│   ├── experiment.py       # Validation tuning, full refits, final_test_run, CV
│   └── results.py          # Summary aggregation, comparison plots, error analysis
├── notebooks/              # Sequential analysis notebooks (with committed outputs)
│   ├── 01_data_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_linear_knn.ipynb
│   ├── 04_rf_svm.ipynb
│   └── 05_results_analysis.ipynb
├── data/splits/            # val_indices.npy (committed fixed validation split)
├── results/                # Benchmark outputs
│   ├── metrics/            # Validation, test, and CV JSON files per experiment
│   ├── figures/            # 300 DPI publication plots (eda_*, features_*, linear_*, results_*, errors_*)
│   ├── summary.csv         # Aggregated 12-experiment benchmark table
│   └── test_log.csv        # Append-only audit log of final test runs
├── docs/                   # Full capstone engineering documentation
│   ├── prd.md              # Requirements and research questions
│   ├── system-design.md    # Architecture and leakage controls
│   ├── decisions.md        # Comprehensive ADR decision log (D-001 to D-031)
│   ├── module-interfaces.md # Function contracts and signatures
│   ├── data-formats.md     # File formats, schemas, and naming conventions
│   ├── progress.md         # Phase tracking and session log
│   └── phases/v1/          # Individual phase specification files
├── CONTRIBUTING.md         # Contributor step-by-step workflow guide
├── AGENTS.md               # Rules for human and AI contributors
├── requirements.txt        # Pinned Python 3.12 dependencies
└── README.md               # Project overview, setup, results, and synthesis
```
