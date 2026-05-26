from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor

from .model import DeepEnergyModel


class LitEP(pl.LightningModule):
    def __init__(
        self,
        *,
        layer_sizes: list[int],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        beta_nudge: float = 0.9,
        free_steps: int = 55,
        nudge_steps: int = 14,
        dt: float = 0.1,
        rho: str = "hardtanh",
        fhn_delta: float = 0.75,
        fhn_epsilon: float = 0.85,
        fhn_alpha: float = 1.08,
        fhn_beta: float = 0.0,
        weight_scale: float = 0.014,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.ep_model = DeepEnergyModel(
            layer_sizes,
            rho=rho,
            fhn_delta=fhn_delta,
            fhn_epsilon=fhn_epsilon,
            fhn_alpha=fhn_alpha,
            fhn_beta=fhn_beta,
            weight_scale=weight_scale,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ep_model.predict(x, steps=self.hparams.free_steps, dt=self.hparams.dt)

    def _classification_metrics(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.ep_model.predict(x, steps=self.hparams.free_steps, dt=self.hparams.dt)
        ce = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return ce, acc

    def training_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        eqprop_objective = self.ep_model.centered_eqprop_loss(
            x,
            y,
            beta_nudge=self.hparams.beta_nudge,
            free_steps=self.hparams.free_steps,
            nudge_steps=self.hparams.nudge_steps,
            dt=self.hparams.dt,
        )
        with torch.no_grad():
            free_ce, train_acc = self._classification_metrics(x, y)

        self.log("train_eqprop_objective", eqprop_objective, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log("train_free_ce", free_ce, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log("train_acc", train_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return eqprop_objective

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        free_ce, val_acc = self._classification_metrics(x, y)
        self.log("val_free_ce", free_ce, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_acc", val_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        return free_ce

    def test_step(self, batch, batch_idx: int) -> Tensor:
        x, y = batch
        free_ce, test_acc = self._classification_metrics(x, y)
        self.log("test_free_ce", free_ce, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("test_acc", test_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        return free_ce

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
