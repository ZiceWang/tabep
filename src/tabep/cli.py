from __future__ import annotations

import sys

import typer

from .train import main as train_main

app = typer.Typer(no_args_is_help=True)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(ctx: typer.Context) -> None:
    """Train with Hydra + PyTorch Lightning.

    Pass Hydra overrides after `train`, e.g.
    `tabep train data.limit_train=256 model.hidden_size=64`.
    """
    _ = ctx
    overrides = sys.argv[2:]
    if overrides and overrides[0] == "mnist":
        overrides = overrides[1:]
    sys.argv = [sys.argv[0], *overrides]
    train_main()


def main() -> None:
    app()
