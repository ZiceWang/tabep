# tabep

> 🤖 **Note for AI agents**: This repository contains a [`.agent/`](.agent/) directory with structured rules, workflows, and a command reference designed for AI consumption and experiment reproduction. If you are an AI agent, consider reading the files under `.agent/` first.

PyTorch TabEP / equilibrium propagation training framework managed with `uv`.

## MNIST smoke test

```powershell
uv run tabep train mnist data.limit_train=256 data.limit_test=128 model.hidden_size=64 model.hidden_layers=2 eqprop.free_steps=5 eqprop.nudge_steps=2 trainer.max_epochs=1 trainer.accelerator=cpu
```

## Full MNIST run

```powershell
uv run tabep train mnist
```

Each experiment writes outputs to `outputs/tabep-mnist/<date>/<time>/`, and Lightning saves checkpoints under the `checkpoints/` directory there.

## Drug200 TabEP benchmark

```powershell
uv run tabep-drug200
```

This script loads `milotix/drug200` from Hugging Face, compares `TabEP`, TileLang-accelerated KNN, TileLang-accelerated RBF kernel classifier, decision tree, and logistic regression on Drug200, then writes accuracy, macro-F1, macro-precision, and macro-recall to `outputs/tabep-drug200/drug200_benchmark.csv`. Ensemble-learning baselines such as RandomForest, ExtraTrees, and gradient boosting are intentionally excluded. It uses a stratified holdout split by default; add `--cv` for more stable 5-fold cross-validation. To use another compatible dataset source, pass `--dataset <huggingface-repo-or-local-csv>`. Add `--tilelang-dynamics --device cuda` to use TileLang kernels for TabEP inference dynamics. TabEP now defaults to the score-first supervised path (`--training-mode gd`, `--free-steps 2`) with trajectory-guided supervision: every relaxation step is trained with a time-weighted classification loss, intermediate logits are regularized toward the final relaxed prediction, and the final state receives a small class-margin penalty. This makes the supervised variant depend on the dynamics trajectory rather than only the last state while keeping inference cheap. Use `--no-trajectory-guidance` to disable this objective, or tune it with `--trajectory-consistency` and `--trajectory-margin`. Use `--mlp-ablation` to also run a plain supervised MLP with approximately the same trainable parameter count as TabEP+readout. Use `--training-mode guided` only when comparing against the EP-guided objective, or `--training-mode ep` for pure EP. Add `--calibrate-readout` to test post-training logistic calibration. Add `--all-training-modes` to run `TabEP-ep`, `TabEP-gd`, and `TabEP-guided` in one benchmark.

## Reproducing reports

The scripts that support the report experiments live in the tracked top-level `scripts/` directory. Generated outputs are written under ignored `outputs/` directories. See [`docs/reproduce.md`](docs/reproduce.md) for commands to reproduce the two Drug200 report PDFs, result tables, and figures.
