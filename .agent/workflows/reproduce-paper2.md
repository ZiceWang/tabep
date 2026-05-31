---
description: Reproduce Paper 2 — Drug200 clustering analysis
globs: *
alwaysApply: false
---

# Reproduce Paper 2: Drug200 Clustering Experiment

## Overview

Paper 2 explores all ten 2-feature combinations of Drug200's five features (`Age`, `Sex`, `BP`, `Cholesterol`, `Na_to_K`) and compares clustering algorithms against the known `Drug` label.

Clustering methods compared:
- **K-means**
- **Fuzzy C-means**
- **DBSCAN**
- **Agglomerative clustering**
- **NSGA-II multi-objective clustering**

## Prerequisites

```bash
cd /data1/neu_lab2/tabep
source .venv/bin/activate
```

## Step 1: Run Clustering Experiment

```bash
uv run python -m tabep.reports.drug200_clustering \
  --dataset data/raw/drug200/drug200.csv \
  --output-dir outputs/paper2-drug200-clustering
```

**Runtime:** ~1-5 minutes on CPU.

**Expected outputs in `outputs/paper2-drug200-clustering/`:**
| File | Description |
|------|-------------|
| `drug200_clustering_metrics.csv` | All clustering results |
| `drug200_best_by_feature_pair.csv` | Best method per feature pair |
| `drug200_method_summary.csv` | Average metrics per method |
| `drug200_clustering_summary.json` | All metrics in JSON |
| `figures/*.pdf` | Per-pair cluster visualizations |

**Metrics include:** NMI (Normalized Mutual Information), ARI (Adjusted Rand Index), silhouette score, etc.

## Step 2: Copy Key Figures to Paper

```bash
cp outputs/paper2-drug200-clustering/figures/method_average_metrics.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/best_feature_pair_nmi.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/BP__Na_to_K__agglomerative.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/Cholesterol__Na_to_K__fuzzy_cmeans.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/Age__Na_to_K__nsga2.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/BP__Cholesterol__kmeans.pdf papers/paper2/graphs/
```

## Step 3: Compile Paper 2 PDF

```bash
cd papers/paper2
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/paper2 main.tex
cp ../build/paper2/main.pdf main.pdf
cd ../..
```

**Requires:** `latexmk` (TeX Live). PDF at `papers/paper2/main.pdf`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: tabep.reports.drug200_clustering` | Run `uv sync` first |
| No clustering figures generated | Check `--output-dir` is writable |
| Dataset not found | Verify `data/raw/drug200/drug200.csv` exists |
| LaTeX compilation fails | Install `texlive-latex-extra` and `texlive-science` |
