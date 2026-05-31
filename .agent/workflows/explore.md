---
description: How to explore and understand the tabep codebase
globs: *
alwaysApply: false
---

# Explore the TabEP Codebase

Use this workflow when you need to understand how TabEP works internally, find specific code, or debug issues.

## Step 1: Understand the High-Level Architecture

Read these files in order:

| Order | File | What it tells you |
|-------|------|-------------------|
| 1 | `README.md` or `README_zh.md` | Project overview and quick start |
| 2 | `docs/reproduce.md` | Full experiment reproduction guide |
| 3 | `pyproject.toml` | Dependencies and CLI entry points |
| 4 | `src/tabep/cli.py` | How CLI commands are wired |
| 5 | `src/tabep/tabular_benchmark.py` (first 80 lines) | Benchmark structure and metrics |

## Step 2: Dive into Core Algorithm

| File | Key classes/functions | Purpose |
|------|-----------------------|---------|
| `src/tabep/model.py` | `DeepEnergyModel` | Core FHN dynamics, EP loss functions, trajectory-guided loss |
| `src/tabep/tabep.py` | `TabEnergyModel`, `LitTabEP` | TabEP-specific overrides and aliases |
| `src/tabep/module.py` | `LitEP` | PyTorch Lightning wrapper (training/val/test steps) |

### Key Methods in `DeepEnergyModel`

```python
step(states, y, beta_nudge, dt)       # Single FHN relaxation step
run_dynamics(x, y, beta_nudge, steps)  # Run full relaxation trajectory
predict(x, steps, dt)                  # Inference: free-run dynamics
centered_eqprop_loss(...)              # Centered EP training objective
supervised_dynamics_loss(...)          # Simple GD supervised loss
trajectory_guided_loss(...)            # GD + trajectory supervision
guided_eqprop_loss(...)                # Hybrid EP + guidance
```

## Step 3: Understand the Readout Heads

| File | Class | Purpose |
|------|-------|---------|
| `src/tabep/tabular_benchmark.py` | `TabEPReadout` | Concatenates state + input, passes through MLP |
| `src/tabep/tabular_benchmark.py` | `PrototypeRBFReadout` | RBF prototype-based readout with energy-state gating |

## Step 4: Understand Data Pipeline

| File | Key Function | Purpose |
|------|-------------|---------|
| `src/tabep/tabular.py` | `load_drug200_bundle()` | Drug200 loading + preprocessing |
| `src/tabep/tabular.py` | `load_ucirepo_bundle()` | UCI dataset loading |
| `src/tabep/tabular.py` | `_make_preprocessing_pipeline()` | One-hot encode categories, standardize numerics |

## Step 5: Understand TileLang Integration

| File | Purpose |
|------|---------|
| `src/tabep/tilelang_dynamics.py` | TileLang CUDA kernels for FHN dynamics |
| `src/tabep/tilelang_classifiers.py` | TileLang-accelerated KNN and RBF SVC |
| `src/tabep/tilelang_utils.py` | Utilities (stderr filtering, etc.) |

TileLang is OPTIONAL. Benchmarks run fine on CPU without it.

## Step 6: Understand Configuration

Hydra configs live in `src/tabep/conf/`:
- `config.yaml` — default config
- `eqprop/` — equilibrium propagation settings
- `model/` — model architecture settings
- `data/` — data settings
- `trainer/` — Lightning trainer settings

Override via command line:
```bash
uv run tabep train mnist model.hidden_size=64 eqprop.free_steps=5
```

## Step 7: Debugging Common Failures

### "CUDA error: out of memory"
- Remove `--tilelang-dynamics --device cuda` flags
- Reduce batch size via Hydra override: `data.batch_size=32`
- Use CPU: `trainer.accelerator=cpu`

### "ImportError: No module named 'tabep'"
```bash
uv sync
# or
pip install -e .
```

### Experiment outputs not found
Check the specific `outputs/` subdirectory. Each command uses `--output-dir` to specify location.

### Metrics look wrong
Check that the benchmark CSV contains the expected 4 metrics: accuracy, macro_f1, macro_precision, macro_recall.

## Key Research Questions to Answer from Results

1. **Does TabEP beat MLP?** Compare macro-F1 in `outputs/tabep-drug200-proto/drug200_benchmark.csv`
2. **Does trajectory guidance help?** Compare `gd` with and without `--no-trajectory-guidance`
3. **How does EP compare to GD?** Use `--all-training-modes` and compare metrics
4. **Is TileLang faster?** Compare runtime with and without `--tilelang-dynamics --device cuda`
5. **How stable are results?** Check `outputs/paper1-reproduction/seed_study_summary.csv`
