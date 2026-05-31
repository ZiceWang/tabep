---
description: Reproduce Paper 1 — Drug200 TabEP benchmark
globs: *
alwaysApply: false
---

# Reproduce Paper 1: Drug200 Classification and TabEP Benchmark

## Overview

Paper 1 evaluates TabEP on the Drug200 dataset, comparing against parameter-matched MLP, TileLang-KNN, TileLang-RBF, decision tree, and logistic regression. It also includes a seed stability study and efficiency analysis.

## Prerequisites

```bash
cd /data1/neu_lab2/tabep
source .venv/bin/activate   # or: uv sync && source .venv/bin/activate
```

## Step 1: Run the Main Drug200 Benchmark

**What it does:** Trains TabEP (Prototype RBF readout) + baselines on Drug200, writes metrics and prediction arrays.

```bash
uv run tabep-drug200 \
  --dataset data/raw/drug200/drug200.csv \
  --output-dir outputs/tabep-drug200-proto \
  --mlp-ablation
```

**Runtime:** ~1-5 minutes on CPU.

**Expected outputs in `outputs/tabep-drug200-proto/`:**
- `drug200_benchmark.csv` — accuracy, macro-F1, macro-precision, macro-recall
- `drug200_benchmark.json` — same metrics in JSON
- `tabep_predictions.npy` — TabEP prediction array
- `mlp_matched_predictions.npy` — parameter-matched MLP predictions
- `tabebm_augmented_knn_predictions.npy`, etc.

**Expected macro-F1 ≈ 0.8389 ± 0.0618 for TabEP.**

### Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `uv sync` |
| `CUDA out of memory` | Omit `--tilelang-dynamics --device cuda`, run on CPU |
| Dataset not found | Check `data/raw/drug200/drug200.csv` exists |
| Wrong metrics | Check `drug200_benchmark.csv` for all 4 metrics |

## Step 2: Seed Stability Study (Optional)

**What it does:** Runs TabEP with 10 different random seeds to measure variance.

```bash
uv run python scripts/run_seed_study.py
```

**Runtime:** ~10-30 minutes.

**Expected outputs in `outputs/paper1-reproduction/`:**
- `seed_study_raw.csv` — per-seed results
- `seed_study_summary.csv` — mean ± std across seeds
- `seed_study_summary.json` — same data in JSON

## Step 3: Efficiency Analysis (Optional)

**What it does:** Measures inference time vs. number of relaxation steps.

```bash
uv run python scripts/compute_efficiency.py
```

**Runtime:** ~2-5 minutes.

**Expected outputs in `outputs/paper1-reproduction/`:**
- `efficiency_results.csv`
- `efficiency_results.json`

## Step 4: Generate Figures

**What it does:** Reads all results and generates Paper 1 figures.

```bash
uv run python scripts/make_paper1_figures.py
```

**Expected outputs:** figures written to `papers/paper1/graphs/`.

**Required inputs (checked by script):**
- `outputs/tabep-drug200-proto/` — from Step 1
- `outputs/uci-suite-proto/` — UCI benchmarks (can be missing; script skips gracefully)
- `outputs/uci-suite-proto-predictions/` — UCI predictions (can be missing)
- `outputs/paper1-reproduction/` — from Steps 2-3 (can be missing)

## Step 5: Compile Paper 1 PDF

```bash
cd papers/paper1
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/paper1 main.tex
cp ../build/paper1/main.pdf main.pdf
cd ../..
```

**Requires:** `latexmk` (TeX Live). The PDF will be at `papers/paper1/main.pdf`.

## Full Pipeline (One-Button Reproduce)

```bash
cd /data1/neu_lab2/tabep
source .venv/bin/activate

# Step 1-5 in sequence
uv run tabep-drug200 --dataset data/raw/drug200/drug200.csv --output-dir outputs/tabep-drug200-proto --mlp-ablation
uv run python scripts/run_seed_study.py
uv run python scripts/compute_efficiency.py
uv run python scripts/make_paper1_figures.py

cd papers/paper1
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/paper1 main.tex
cp ../build/paper1/main.pdf main.pdf
cd ../..
```

## UCI Suite Benchmarks (for Complete Reproduction)

To also reproduce UCI results referenced in the paper, run the benchmark for each dataset. These are more time-consuming.

The benchmark IDs are:
- `adult` (ID 2), `breast-cancer-wisconsin-diagnostic` (17), `covertype` (31), `iris` (53), `wine` (109)
- `letter-recognition` (59), `optdigits` (80), `pendigits` (81), `satimage` (146), `segment` (50)
- `shuttle` (148), `vehicle-silhouettes` (149), `vowel-recognition` (104)

Each dataset can be run via the tabep-drug200 command with `--dataset <name>` using the Hugging Face / ucimlrepo loader.
