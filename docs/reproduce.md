# Reproducing the Two Drug200 Reports

This document records the commands used to reproduce the experiments, figures, and PDFs for the two reports. Run commands from the repository root unless noted otherwise.

## Environment

Use the project virtual environment or `uv` environment. The repository is expected to be installable in editable/development mode.

```bash
uv sync
```

If using the existing local virtual environment directly:

```bash
source .venv/bin/activate
```

Generated experiment outputs are written under `outputs/`, which is intentionally ignored by Git. Report PDFs live under `papers/paper1/main.pdf` and `papers/paper2/main.pdf`; the `papers/` directory is also ignored, so report source snapshots must be shared separately if needed.

## Paper 1: Drug200 Classification and TabEP Benchmark

### Main Drug200 benchmark

```bash
uv run tabep-drug200 --dataset data/raw/drug200/drug200.csv --output-dir outputs/tabep-drug200-proto --mlp-ablation
```

This produces the main Drug200 classification metrics and prediction arrays used by the first report.

### Optional seed study

```bash
uv run python scripts/run_seed_study.py
```

Outputs:

- `outputs/paper1-reproduction/seed_study_raw.csv`
- `outputs/paper1-reproduction/seed_study_summary.csv`
- `outputs/paper1-reproduction/seed_study_summary.json`

### Optional efficiency study

```bash
uv run python scripts/compute_efficiency.py
```

Outputs:

- `outputs/paper1-reproduction/efficiency_results.csv`
- `outputs/paper1-reproduction/efficiency_results.json`

### UCI suite outputs

The first report also references UCI-suite summaries under `outputs/uci-suite-proto/` and prediction files under `outputs/uci-suite-proto-predictions/`. If these outputs are missing, rerun the repository UCI benchmark command used for the project before regenerating figures.

### Paper 1 figures

```bash
uv run python scripts/make_paper1_figures.py
```

The figure script reads results from:

- `outputs/tabep-drug200-proto/`
- `outputs/uci-suite-proto/`
- `outputs/uci-suite-proto-predictions/`
- `outputs/paper1-reproduction/`

and writes selected figures into `papers/paper1/graphs/`.

### Compile Paper 1

```bash
cd papers/paper1
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/paper1 main.tex
cp ../build/paper1/main.pdf main.pdf
cd ../..
```

## Paper 2: Drug200 Clustering Experiment

Paper 2 uses all ten two-feature combinations from:

- `Age`
- `Sex`
- `BP`
- `Cholesterol`
- `Na_to_K`

and compares the clusters with the known `Drug` label after clustering.

### Run clustering experiment

```bash
uv run python -m tabep.reports.drug200_clustering \
  --dataset data/raw/drug200/drug200.csv \
  --output-dir outputs/paper2-drug200-clustering
```

Outputs:

- `outputs/paper2-drug200-clustering/drug200_clustering_metrics.csv`
- `outputs/paper2-drug200-clustering/drug200_best_by_feature_pair.csv`
- `outputs/paper2-drug200-clustering/drug200_method_summary.csv`
- `outputs/paper2-drug200-clustering/drug200_clustering_summary.json`
- `outputs/paper2-drug200-clustering/figures/*.pdf`

The experiment implements K-means, fuzzy C-means, DBSCAN, agglomerative clustering, and NSGA-II-style multi-objective clustering in project code.

### Copy selected Paper 2 figures

```bash
cp outputs/paper2-drug200-clustering/figures/method_average_metrics.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/best_feature_pair_nmi.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/BP__Na_to_K__agglomerative.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/Cholesterol__Na_to_K__fuzzy_cmeans.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/Age__Na_to_K__nsga2.pdf papers/paper2/graphs/
cp outputs/paper2-drug200-clustering/figures/BP__Cholesterol__kmeans.pdf papers/paper2/graphs/
```

### Compile Paper 2

```bash
cd papers/paper2
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/paper2 main.tex
cp ../build/paper2/main.pdf main.pdf
cd ../..
```

## Source Code Location

The public source repository is:

<https://github.com/ZiceWang/tabep>
