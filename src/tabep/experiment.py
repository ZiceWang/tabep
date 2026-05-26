from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import SwanLabLogger
except ImportError:  # pragma: no cover
    SwanLabLogger = None  # type: ignore[assignment]

from .data import mnist_loaders
from .module import LitEP


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def build_model(cfg: DictConfig) -> LitEP:
    layer_sizes = [
        int(cfg.model.input_size),
        *([int(cfg.model.hidden_size)] * int(cfg.model.hidden_layers)),
        int(cfg.model.output_size),
    ]
    return LitEP(
        layer_sizes=layer_sizes,
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        beta_nudge=float(cfg.eqprop.beta_nudge),
        free_steps=int(cfg.eqprop.free_steps),
        nudge_steps=int(cfg.eqprop.nudge_steps),
        dt=float(cfg.eqprop.dt),
        rho=str(cfg.model.rho),
        fhn_delta=float(cfg.model.fhn_delta),
        fhn_epsilon=float(cfg.model.fhn_epsilon),
        fhn_alpha=float(cfg.model.fhn_alpha),
        fhn_beta=float(cfg.model.fhn_beta),
        weight_scale=float(cfg.model.weight_scale),
    )


def build_logger(cfg: DictConfig):
    loggers = []
    if cfg.logging.csv.enabled:
        loggers.append(CSVLogger(save_dir=str(cfg.paths.output_dir), name="csv", version=""))
    if cfg.logging.swanlab.enabled:
        if SwanLabLogger is None:
            raise RuntimeError("SwanLabLogger is unavailable. Check pytorch-lightning/swanlab versions.")
        loggers.append(
            SwanLabLogger(
                project=str(cfg.logging.swanlab.project),
                experiment_name=str(cfg.logging.swanlab.experiment_name),
                save_dir=str(cfg.paths.output_dir),
            )
        )
    if not loggers:
        return False
    return loggers if len(loggers) > 1 else loggers[0]


def build_callbacks(cfg: DictConfig):
    callbacks = []
    if cfg.checkpoint.enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(cfg.paths.checkpoint_dir),
                filename=str(cfg.checkpoint.filename),
                monitor=str(cfg.checkpoint.monitor),
                mode=str(cfg.checkpoint.mode),
                save_last=bool(cfg.checkpoint.save_last),
                save_top_k=int(cfg.checkpoint.save_top_k),
                auto_insert_metric_name=False,
            )
        )
    if cfg.logging.lr_monitor:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if cfg.logging.rich_progress:
        callbacks.append(RichProgressBar())
    return callbacks


def resolve_accelerator(accelerator: str) -> str:
    if accelerator == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    return accelerator


@hydra.main(version_base=None, config_path="conf", config_name="mnist")
def main(cfg: DictConfig) -> None:
    if cfg.performance.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(str(cfg.performance.float32_matmul_precision))

    pl.seed_everything(int(cfg.seed), workers=True)
    print(OmegaConf.to_yaml(cfg))

    train_loader, val_loader = mnist_loaders(
        batch_size=int(cfg.data.batch_size),
        eval_batch_size=int(cfg.data.eval_batch_size),
        limit_train=cfg.data.limit_train,
        limit_test=cfg.data.limit_test,
        num_workers=int(cfg.data.num_workers),
    )

    model = build_model(cfg)
    trainer = pl.Trainer(
        default_root_dir=str(cfg.paths.output_dir),
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=resolve_accelerator(str(cfg.trainer.accelerator)),
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        limit_train_batches=_as_optional_float(cfg.trainer.limit_train_batches),
        limit_val_batches=_as_optional_float(cfg.trainer.limit_val_batches),
        deterministic=bool(cfg.trainer.deterministic),
        enable_checkpointing=bool(cfg.checkpoint.enabled),
        logger=build_logger(cfg),
        callbacks=build_callbacks(cfg),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
    )

    ckpt_path = cfg.resume.ckpt_path if cfg.resume.ckpt_path else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    if cfg.test_after_fit:
        best_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
        trainer.test(model, dataloaders=val_loader, ckpt_path=best_path or None)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
