---
description: TabEP project overview and agent conventions
globs: *
alwaysApply: true
---

# TabEP — AI Agent Rules

## What is TabEP?

TabEP (**Tab**ular **E**quilibrium **P**ropagation) is a neural classification framework for tabular data. It uses **energy-based dynamical relaxation** — inspired by FitzHugh-Nagumo reaction-diffusion systems — to make predictions by letting neural states settle to equilibrium. Training can use pure **Equilibrium Propagation (EP)**, standard **backprop (GD)**, or a hybrid **guided** approach.

## Key Facts

| Fact | Value |
|------|-------|
| Python | ≥ 3.13 |
| Package manager | `uv` |
| PyTorch | 2.11 (CUDA 12.8) |
| GPU kernels | TileLang (optional, JIT-compiled CUDA) |
| Training modes | `ep`, `gd`, `guided` (default: `gd`) |
| Experiment tracking | SwanLab (alternative to wandb, works in China) |
| Config | Hydra |
| CLI | Typer |

## Project Structure

```
tabep/
├── main.py                         # entry: calls tabep.cli.main()
├── pyproject.toml                  # uv-managed, pip install . works
├── .agent/                         # <-- YOU ARE HERE
├── docs/reproduce.md               # full reproduction commands
├── data/raw/drug200/               # Drug200 dataset (CSV)
├── outputs/                        # experiment results (git-ignored)
├── scripts/                        # reproduction & analysis scripts
│   ├── compute_efficiency.py
│   ├── make_paper1_figures.py
│   └── run_seed_study.py
├── papers/                         # LaTeX source + PDFs (git-ignored)
└── src/tabep/
    ├── cli.py                      # CLI entry (Typer)
    ├── train.py                    # Hydra + PL training
    ├── model.py                    # DeepEnergyModel (FHN + centered EP)
    ├── module.py                   # PyTorch Lightning wrapper
    ├── tabep.py                    # TabEnergyModel alias
    ├── tabular.py                  # data loading & preprocessing
    ├── tabular_benchmark.py        # main benchmark engine
    ├── data.py / experiment.py / lit.py
    ├── tilelang_dynamics.py        # TileLang CUDA kernels
    ├── tilelang_classifiers.py     # TileLang KNN / RBF
    ├── conf/                       # Hydra configs
    └── reports/                    # report generators
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `uv run tabep train [overrides]` | Hydra + PL training (use `mnist` for smoke test) |
| `uv run tabep-drug200 [options]` | Drug200 benchmark (TabEP vs baselines) |
| `uv run tabep-drug200-clustering` | Drug200 feature-pair clustering analysis |
| `uv run python scripts/run_seed_study.py` | Seed stability study |
| `uv run python scripts/compute_efficiency.py` | Efficiency analysis |
| `uv run python scripts/make_paper1_figures.py` | Generate Paper 1 figures |

## Critical Rules for AI Agents

### 1. Always use `uv run` to execute commands
- The project uses `uv` for package management.
- **Never** run `python` directly without `uv run` prefix, unless the venv is already activated.
- Exception: scripts inside `papers/` (LaTeX compilation) don't need `uv run`.

### 2. Activate venv first if running multiple commands
```bash
source .venv/bin/activate
```
Then subsequent `uv run` or `python` commands will use the correct environment.

### 3. Output directories are git-ignored
All experiment outputs go under `outputs/` (git-ignored). Don't worry about cluttering the repo.

### 4. Papers directory is git-ignored
`papers/` contains LaTeX source and build artifacts. Report PDFs are NOT in the repo.

### 5. Default training mode is `gd` (score-first supervised)
- `--training-mode gd` with `--free-steps 2` is the current default.
- Trajectory-guided supervision is enabled by default.
- Use `--all-training-modes` to compare all three modes.

### 6. TileLang is optional
- Pass `--tilelang-dynamics --device cuda` to use TileLang-accelerated CUDA kernels.
- Without these flags, the benchmark runs on CPU using PyTorch.

### 7. Hardware awareness
- The project GPU server (NEU-lab2) has CUDA 12.8 available.
- For Drug200 benchmarks, CPU-only is fine (~1-5 min).
- For UCI suite benchmarks, GPU is recommended for larger datasets (e.g. covertype, shuttle).

### 8. If commands fail with import errors
Run `uv sync` first to ensure all dependencies are installed.

## Training Modes Quick Reference

| Mode | Training Signal | Inference | Use Case |
|------|----------------|-----------|----------|
| `ep` | Centered EP (no backprop) | Free-run dynamics | Biologically plausible learning |
| `gd` | BPTT + trajectory guidance | Free-run dynamics | **Default** — best accuracy |
| `guided` | EP objective + guidance | Free-run dynamics | Comparing against pure EP |

## Default Hyperparameters (GD mode)

| Param | Default | Description |
|-------|---------|-------------|
| `free_steps` | 2 (GD); 55 (EP) | Relaxation steps during training |
| `nudge_steps` | 14 | Nudge phase steps (EP mode) |
| `dt` | 0.1 | Time step |
| `lr` | 1e-3 | Learning rate |
| `trajectory_consistency` | 0.1 | KL consistency weight |
| `trajectory_margin` | 0.05 | Margin penalty weight |
