from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .model import DeepEnergyModel
from .tabular import load_drug200_bundle, load_ucirepo_bundle
from .tilelang_classifiers import TileLangKNeighborsClassifier, TileLangRbfSVC
from .tilelang_dynamics import run_tilelang_dynamics
from .tilelang_utils import install_tilelang_stderr_filter


METRIC_NAMES = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
TABEP_TRAINING_MODES = ["ep", "gd", "guided"]
UCI_DATASETS = {
    "adult": 2,
    "breast-cancer-wisconsin-diagnostic": 17,
    "covertype": 31,
    "iris": 53,
    "letter-recognition": 59,
    "optdigits": 80,
    "pendigits": 81,
    "vowel-recognition": 104,
    "wine": 109,
    "satimage": 146,
    "shuttle": 148,
    "vehicle-silhouettes": 149,
    "segment": 50,
}

BENCHMARK_DATASETS = [
    ("adult", 2),
    ("breast-cancer-wisconsin-diagnostic", 17),
    ("covertype", 31),
    ("iris", 53),
    ("wine", 109),
    ("letter-recognition", 59),
    ("optdigits", 80),
    ("pendigits", 81),
    ("satimage", 146),
    ("segment", 50),
    ("shuttle", 148),
    ("vehicle-silhouettes", 149),
    ("vowel-recognition", 104),
]


class TabEPReadout(nn.Module):
    """Small supervised head used when the goal is maximum tabular accuracy."""

    def __init__(self, input_dim: int, state_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, x], dim=1))


class PrototypeRBFReadout(nn.Module):
    """Fast one-step readout combining energy-state gating and RBF prototypes."""

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        prototypes_per_class: int = 2,
        hybrid_linear: bool = True,
        fusion_type: str = "concat",
        interaction_rank: int = 8,
    ) -> None:
        super().__init__()
        feature_dim = max(hidden_dim, 32)
        self.output_dim = output_dim
        self.prototypes_per_class = prototypes_per_class
        self.hybrid_linear = hybrid_linear
        self.fusion_type = fusion_type
        self.interaction_rank = interaction_rank
        if fusion_type == "concat":
            self.concat_encoder = nn.Sequential(
                nn.Linear(input_dim + state_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
            )
        elif fusion_type == "gate":
            self.x_encoder = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU(),
            )
            self.state_gate = nn.Sequential(
                nn.Linear(state_dim, feature_dim),
                nn.Sigmoid(),
            )
        elif fusion_type in {"bilinear", "concat-bilinear"}:
            rank = max(1, interaction_rank)
            self.x_encoder = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU())
            self.state_encoder = nn.Sequential(nn.Linear(state_dim, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU())
            self.x_factors = nn.Linear(input_dim, rank, bias=False)
            self.state_factors = nn.Linear(state_dim, rank, bias=False)
            self.interaction_encoder = nn.Sequential(nn.Linear(rank, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU())
            if fusion_type == "concat-bilinear":
                self.concat_encoder = nn.Sequential(
                    nn.Linear(input_dim + state_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.SiLU(),
                )
                self.mix = nn.Sequential(nn.Linear(feature_dim * 4, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU())
            else:
                self.mix = nn.Sequential(nn.Linear(feature_dim * 3, feature_dim), nn.LayerNorm(feature_dim), nn.SiLU())
        else:
            raise ValueError(f"Unsupported fusion_type={fusion_type!r}.")
        self.feature = nn.Sequential(
            nn.Dropout(0.03),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )
        self.prototypes = nn.Parameter(0.05 * torch.randn(output_dim, prototypes_per_class, feature_dim))
        self.log_gamma = nn.Parameter(torch.tensor(0.0))
        self.prototype_scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.linear = nn.Linear(feature_dim, output_dim) if hybrid_linear else None
        if self.linear is not None:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            fused = self.concat_encoder(torch.cat([state, x], dim=1))
        elif self.fusion_type == "gate":
            fused = self.x_encoder(x) * (1.0 + self.state_gate(state))
        elif self.fusion_type == "bilinear":
            x_latent = self.x_encoder(x)
            state_latent = self.state_encoder(state)
            interaction = self.interaction_encoder(self.x_factors(x) * self.state_factors(state))
            fused = self.mix(torch.cat([x_latent, state_latent, interaction], dim=1))
        elif self.fusion_type == "concat-bilinear":
            concat_latent = self.concat_encoder(torch.cat([state, x], dim=1))
            x_latent = self.x_encoder(x)
            state_latent = self.state_encoder(state)
            interaction = self.interaction_encoder(self.x_factors(x) * self.state_factors(state))
            fused = self.mix(torch.cat([concat_latent, x_latent, state_latent, interaction], dim=1))
        else:  # pragma: no cover - guarded in __init__.
            raise RuntimeError(f"Unsupported fusion_type={self.fusion_type!r}.")
        z = F.normalize(self.feature(fused), dim=1)
        prototypes = F.normalize(self.prototypes, dim=2)
        dist2 = (z[:, None, None, :] - prototypes[None, :, :, :]).square().sum(dim=3)
        gamma = F.softplus(self.log_gamma) + 1e-4
        proto_logits = torch.logsumexp(-gamma * dist2, dim=2) + self.bias
        proto_logits = F.softplus(self.prototype_scale) * proto_logits
        if self.linear is None:
            return proto_logits
        return self.linear(z) + proto_logits


class MLPClassifier(nn.Module):
    """Plain supervised MLP for parameter-count matched ablation."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int, dropout: float = 0.05) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def readout_logits(readout: nn.Module | None, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if readout is None:
        return state
    if isinstance(readout, (TabEPReadout, PrototypeRBFReadout)):
        return readout(state, x)
    return readout(state)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def matched_mlp_width(input_dim: int, output_dim: int, hidden_layers: int, target_params: int) -> int:
    best_width = 4
    best_gap = float("inf")
    for width in range(4, 513):
        candidate = MLPClassifier(input_dim, width, output_dim, hidden_layers)
        gap = abs(count_parameters(candidate) - target_params)
        if gap < best_gap:
            best_width = width
            best_gap = gap
    return best_width


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return requested


def maybe_tqdm(iterable, *, enabled: bool, **kwargs):
    return tqdm(iterable, **kwargs) if enabled else iterable


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _safe_cv_splits(y: np.ndarray, requested: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(requested, int(counts.min())))


def cross_val_predict_manual(model, x: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    predictions = np.empty_like(y)
    for train_idx, test_idx in cv.split(x, y):
        fold_model = clone(model)
        fold_model.fit(x[train_idx], y[train_idx])
        predictions[test_idx] = fold_model.predict(x[test_idx])
    return predictions


def run_baselines(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cv_splits: int,
    device: torch.device,
    progress: bool,
) -> dict[str, dict[str, float]]:
    models = {
        "TileLang-KNN": TileLangKNeighborsClassifier(n_neighbors=5, weights="distance", device=device),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=seed),
        "TileLang-RBF-SVM": TileLangRbfSVC(C=3.0, gamma="scale", class_weight="balanced", device=device),
        "LogisticRegression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
    }
    cv = StratifiedKFold(n_splits=_safe_cv_splits(y, cv_splits), shuffle=True, random_state=seed)
    results: dict[str, dict[str, float]] = {}
    for name, model in maybe_tqdm(models.items(), enabled=progress, desc="sklearn CV baselines", unit="model"):
        y_pred = cross_val_predict_manual(model, x, y, cv)
        results[name] = compute_metrics(y, y_pred)
    return results


def run_baselines_holdout(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    progress: bool,
) -> dict[str, dict[str, float]]:
    models = {
        "TileLang-KNN": TileLangKNeighborsClassifier(n_neighbors=5, weights="distance", device=device),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=seed),
        "TileLang-RBF-SVM": TileLangRbfSVC(C=3.0, gamma="scale", class_weight="balanced", device=device),
        "LogisticRegression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
    }
    results: dict[str, dict[str, float]] = {}
    for name, model in maybe_tqdm(models.items(), enabled=progress, desc="sklearn holdout baselines", unit="model"):
        model.fit(x_train, y_train)
        results[name] = compute_metrics(y_test, model.predict(x_test))
    return results


def make_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
    weights = y.numel() / (num_classes * counts)
    return weights / weights.mean()


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weights = {int(cls): y.size / (len(classes) * count) for cls, count in zip(classes, counts)}
    return np.asarray([weights[int(label)] for label in y], dtype=np.float64)


def fit_weighted_logistic_blender(train_logits: np.ndarray, y_train: np.ndarray, seed: int) -> LogisticRegression:
    blender = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=5.0,
        solver="lbfgs",
        random_state=seed,
    )
    blender.fit(train_logits, y_train)
    return blender


def make_readout(
    input_dim: int,
    state_dim: int,
    output_dim: int,
    hidden_dim: int,
    device: torch.device,
    *,
    readout_type: str = "mlp",
    prototypes_per_class: int = 2,
    prototype_fusion: str = "concat",
    interaction_rank: int = 8,
) -> nn.Module:
    if readout_type == "mlp":
        readout = TabEPReadout(input_dim, state_dim, output_dim, max(hidden_dim, 32)).to(device)
        final = readout.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
    elif readout_type == "prototype":
        readout = PrototypeRBFReadout(
            input_dim,
            state_dim,
            output_dim,
            max(hidden_dim, 32),
            prototypes_per_class=prototypes_per_class,
            hybrid_linear=False,
            fusion_type=prototype_fusion,
            interaction_rank=interaction_rank,
        ).to(device)
    elif readout_type == "hybrid-prototype":
        readout = PrototypeRBFReadout(
            input_dim,
            state_dim,
            output_dim,
            max(hidden_dim, 32),
            prototypes_per_class=prototypes_per_class,
            hybrid_linear=True,
            fusion_type=prototype_fusion,
            interaction_rank=interaction_rank,
        ).to(device)
    else:
        raise ValueError(f"Unsupported readout_type={readout_type!r}.")
    return readout


def training_loss(
    model: DeepEnergyModel,
    readout: nn.Module | None,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    training_mode: str,
    beta_nudge: float,
    free_steps: int,
    nudge_steps: int,
    dt: float,
    guidance_weight: float,
    guidance_beta: float,
    class_weights: torch.Tensor,
    hidden_state_l2: float,
    trajectory_guidance: bool = True,
    trajectory_consistency: float = 0.1,
    trajectory_margin: float = 0.05,
    prototype_weight: float = 0.0,
) -> torch.Tensor:
    if training_mode == "ep":
        return model.centered_eqprop_loss(
            x_batch,
            y_batch,
            beta_nudge=beta_nudge,
            free_steps=free_steps,
            nudge_steps=nudge_steps,
            dt=dt,
        )
    if training_mode == "gd":
        if trajectory_guidance:
            loss = model.trajectory_guided_loss(
                x_batch,
                y_batch,
                steps=free_steps,
                dt=dt,
                readout=readout,
                class_weights=class_weights,
                hidden_state_l2=hidden_state_l2,
                consistency_weight=trajectory_consistency,
                margin_weight=trajectory_margin,
                target_beta=guidance_beta,
            )
        else:
            loss = model.supervised_dynamics_loss(
            x_batch,
            y_batch,
            steps=free_steps,
            dt=dt,
            target_beta=guidance_beta,
            readout=readout,
            class_weights=class_weights,
            hidden_state_l2=hidden_state_l2,
            )
        if prototype_weight != 0.0 and isinstance(readout, PrototypeRBFReadout):
            prototypes = F.normalize(readout.prototypes, dim=2)
            flat = prototypes.reshape(-1, prototypes.shape[-1])
            similarity = flat @ flat.T
            mask = ~torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
            separation = similarity[mask].square().mean()
            loss = loss + prototype_weight * separation
        return loss
    if training_mode == "guided":
        return model.guided_eqprop_loss(
            x_batch,
            y_batch,
            beta_nudge=beta_nudge,
            free_steps=free_steps,
            nudge_steps=nudge_steps,
            dt=dt,
            guidance_weight=guidance_weight,
            guidance_beta=guidance_beta,
            readout=readout,
            class_weights=class_weights,
            hidden_state_l2=hidden_state_l2,
        )
    raise ValueError(f"Unsupported training mode: {training_mode}")


@torch.no_grad()
def collect_logits(
    model: DeepEnergyModel,
    readout: nn.Module | None,
    x: torch.Tensor,
    *,
    steps: int,
    dt: float,
    batch_size: int,
    tilelang_dynamics: bool,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        x_batch = x[start : start + batch_size]
        if tilelang_dynamics:
            state = run_tilelang_dynamics(model, x_batch, steps=steps, dt=dt)[-1]
        else:
            state = model.predict(x_batch, steps=steps, dt=dt)
        logits = readout_logits(readout, state, x_batch)
        chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_tabep_holdout(
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
    y_train_np: np.ndarray,
    y_test_np: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    hidden_layers: int,
    free_steps: int,
    nudge_steps: int,
    dt: float,
    beta_nudge: float,
    lr: float,
    weight_decay: float,
    training_mode: str,
    guidance_weight: float,
    guidance_beta: float,
    hidden_state_l2: float,
    readout_guidance: bool,
    calibrate_readout: bool,
    trajectory_guidance: bool = True,
    trajectory_consistency: float = 0.1,
    trajectory_margin: float = 0.05,
    readout_type: str,
    prototypes_per_class: int,
    prototype_fusion: str,
    interaction_rank: int,
    prototype_weight: float,
    device: torch.device,
    tilelang_dynamics: bool,
    progress: bool,
) -> tuple[dict[str, float], np.ndarray]:
    set_seed(seed)
    x_train = torch.from_numpy(x_train_np.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train_np.astype(np.int64)).to(device)
    x_test = torch.from_numpy(x_test_np.astype(np.float32)).to(device)
    layer_sizes = [x_train_np.shape[1], *([hidden_size] * hidden_layers), int(np.unique(y_train_np).size)]
    model = DeepEnergyModel(
        layer_sizes,
        rho="hardtanh",
        weight_scale=0.035,
        fhn_delta=0.75,
        fhn_epsilon=0.35,
        fhn_alpha=0.75,
    ).to(device)
    readout = make_readout(
        x_train_np.shape[1],
        layer_sizes[-1],
        layer_sizes[-1],
        hidden_size,
        device,
        readout_type=readout_type,
        prototypes_per_class=prototypes_per_class,
        prototype_fusion=prototype_fusion,
        interaction_rank=interaction_rank,
    ) if readout_guidance else None
    params = list(model.parameters()) + (list(readout.parameters()) if readout is not None else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    class_weights = make_class_weights(y_train, layer_sizes[-1])
    indices = torch.arange(x_train.shape[0], device=device)
    for epoch in maybe_tqdm(range(epochs), enabled=progress, desc="TabEP holdout", unit="epoch"):
        perm = indices[torch.randperm(indices.numel(), device=device)]
        losses = []
        for start in range(0, perm.numel(), batch_size):
            batch_idx = perm[start : start + batch_size]
            loss = training_loss(
                model,
                readout,
                x_train[batch_idx],
                y_train[batch_idx],
                training_mode=training_mode,
                beta_nudge=beta_nudge,
                free_steps=free_steps,
                nudge_steps=nudge_steps,
                dt=dt,
                guidance_weight=guidance_weight,
                guidance_beta=guidance_beta,
                class_weights=class_weights,
                hidden_state_l2=hidden_state_l2,
                trajectory_guidance=trajectory_guidance,
                trajectory_consistency=trajectory_consistency,
                trajectory_margin=trajectory_margin,
                prototype_weight=prototype_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            message = f"TabEP epoch {epoch + 1:03d}/{epochs} loss={np.mean(losses):.4f}"
            if progress:
                tqdm.write(message)
            else:
                print(message, flush=True)

    train_logits = collect_logits(model, readout, x_train, steps=free_steps, dt=dt, batch_size=batch_size, tilelang_dynamics=tilelang_dynamics)
    test_logits = collect_logits(model, readout, x_test, steps=free_steps, dt=dt, batch_size=batch_size, tilelang_dynamics=tilelang_dynamics)
    if calibrate_readout and training_mode == "gd":
        blender = fit_weighted_logistic_blender(train_logits, y_train_np, seed)
        y_pred = blender.predict(test_logits)
    else:
        y_pred = test_logits.argmax(axis=1)
    return compute_metrics(y_test_np, y_pred), y_pred


def train_mlp_holdout(
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
    y_train_np: np.ndarray,
    y_test_np: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    hidden_layers: int,
    output_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    progress: bool,
) -> tuple[dict[str, float], np.ndarray]:
    set_seed(seed)
    input_dim = x_train_np.shape[1]
    tabep_probe = DeepEnergyModel(
        [input_dim, *([hidden_size] * hidden_layers), output_size],
        rho="hardtanh",
        weight_scale=0.035,
        fhn_delta=0.75,
        fhn_epsilon=0.35,
        fhn_alpha=0.75,
    )
    readout_probe = make_readout(input_dim, output_size, output_size, hidden_size, torch.device("cpu"))
    target_params = count_parameters(tabep_probe) + count_parameters(readout_probe)
    mlp_width = matched_mlp_width(input_dim, output_size, hidden_layers, target_params)
    model = MLPClassifier(input_dim, mlp_width, output_size, hidden_layers).to(device)

    x_train = torch.from_numpy(x_train_np.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train_np.astype(np.int64)).to(device)
    x_test = torch.from_numpy(x_test_np.astype(np.float32)).to(device)
    class_weights = make_class_weights(y_train, output_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    indices = torch.arange(x_train.shape[0], device=device)

    for epoch in maybe_tqdm(range(epochs), enabled=progress, desc=f"MLP holdout ({count_parameters(model)} params)", unit="epoch"):
        perm = indices[torch.randperm(indices.numel(), device=device)]
        losses = []
        for start in range(0, perm.numel(), batch_size):
            batch_idx = perm[start : start + batch_size]
            loss = F.cross_entropy(model(x_train[batch_idx]), y_train[batch_idx], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            message = f"MLP epoch {epoch + 1:03d}/{epochs} loss={np.mean(losses):.4f} width={mlp_width} params={count_parameters(model)} target={target_params}"
            if progress:
                tqdm.write(message)
            else:
                print(message, flush=True)

    with torch.no_grad():
        y_pred = model(x_test).argmax(dim=1).cpu().numpy()
    return compute_metrics(y_test_np, y_pred), y_pred


def train_tabep_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cv_splits: int,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    hidden_layers: int,
    free_steps: int,
    nudge_steps: int,
    dt: float,
    beta_nudge: float,
    lr: float,
    weight_decay: float,
    training_mode: str,
    guidance_weight: float,
    guidance_beta: float,
    hidden_state_l2: float,
    readout_guidance: bool,
    calibrate_readout: bool,
    trajectory_guidance: bool,
    trajectory_consistency: float,
    trajectory_margin: float,
    readout_type: str,
    prototypes_per_class: int,
    prototype_fusion: str,
    interaction_rank: int,
    prototype_weight: float,
    device: torch.device,
    tilelang_dynamics: bool,
    progress: bool,
) -> tuple[dict[str, float], np.ndarray]:
    cv = StratifiedKFold(n_splits=_safe_cv_splits(y, cv_splits), shuffle=True, random_state=seed)
    predictions = np.empty_like(y)

    splits = list(cv.split(x, y))
    for fold_idx, (train_idx, test_idx) in maybe_tqdm(enumerate(splits, start=1), enabled=progress, desc="TabEP CV", total=len(splits), unit="fold"):
        set_seed(seed + fold_idx)
        x_train = torch.from_numpy(x[train_idx].astype(np.float32)).to(device)
        y_train = torch.from_numpy(y[train_idx].astype(np.int64)).to(device)
        x_test = torch.from_numpy(x[test_idx].astype(np.float32)).to(device)
        layer_sizes = [x.shape[1], *([hidden_size] * hidden_layers), int(np.unique(y).size)]
        model = DeepEnergyModel(
            layer_sizes,
            rho="hardtanh",
            weight_scale=0.035,
            fhn_delta=0.75,
            fhn_epsilon=0.35,
            fhn_alpha=0.75,
        ).to(device)
        readout = make_readout(
            x.shape[1],
            layer_sizes[-1],
            layer_sizes[-1],
            hidden_size,
            device,
            readout_type=readout_type,
            prototypes_per_class=prototypes_per_class,
            prototype_fusion=prototype_fusion,
            interaction_rank=interaction_rank,
        ) if readout_guidance else None
        params = list(model.parameters()) + (list(readout.parameters()) if readout is not None else [])
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        class_weights = make_class_weights(y_train, layer_sizes[-1])

        indices = torch.arange(x_train.shape[0], device=device)
        for _ in maybe_tqdm(range(epochs), enabled=progress, desc=f"TabEP fold {fold_idx}", unit="epoch", leave=False):
            perm = indices[torch.randperm(indices.numel(), device=device)]
            for start in range(0, perm.numel(), batch_size):
                batch_idx = perm[start : start + batch_size]
                loss = training_loss(
                    model,
                    readout,
                    x_train[batch_idx],
                    y_train[batch_idx],
                    training_mode=training_mode,
                    beta_nudge=beta_nudge,
                    free_steps=free_steps,
                    nudge_steps=nudge_steps,
                    dt=dt,
                    guidance_weight=guidance_weight,
                    guidance_beta=guidance_beta,
                    class_weights=class_weights,
                    hidden_state_l2=hidden_state_l2,
                    trajectory_guidance=trajectory_guidance,
                    trajectory_consistency=trajectory_consistency,
                    trajectory_margin=trajectory_margin,
                    prototype_weight=prototype_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=2.0)
                optimizer.step()

        train_logits = collect_logits(model, readout, x_train, steps=free_steps, dt=dt, batch_size=batch_size, tilelang_dynamics=tilelang_dynamics)
        test_logits = collect_logits(model, readout, x_test, steps=free_steps, dt=dt, batch_size=batch_size, tilelang_dynamics=tilelang_dynamics)
        if calibrate_readout and training_mode == "gd":
            blender = fit_weighted_logistic_blender(train_logits, y[train_idx], seed + fold_idx)
            predictions[test_idx] = blender.predict(test_logits)
        else:
            predictions[test_idx] = test_logits.argmax(axis=1)

    return compute_metrics(y, predictions), predictions


def train_mlp_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cv_splits: int,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    hidden_layers: int,
    output_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    progress: bool,
) -> tuple[dict[str, float], np.ndarray]:
    cv = StratifiedKFold(n_splits=_safe_cv_splits(y, cv_splits), shuffle=True, random_state=seed)
    predictions = np.empty_like(y)

    splits = list(cv.split(x, y))
    for fold_idx, (train_idx, test_idx) in maybe_tqdm(enumerate(splits, start=1), enabled=progress, desc="MLP CV", total=len(splits), unit="fold"):
        metrics, fold_pred = train_mlp_holdout(
            x[train_idx],
            x[test_idx],
            y[train_idx],
            y[test_idx],
            seed=seed + fold_idx,
            epochs=epochs,
            batch_size=batch_size,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            progress=False,
        )
        predictions[test_idx] = fold_pred
        if progress:
            tqdm.write(f"MLP fold {fold_idx} macro_f1={metrics['macro_f1']:.4f}")

    return compute_metrics(y, predictions), predictions


def dataframe_from_results(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(results, orient="index")
    df = df[METRIC_NAMES]
    df.insert(0, "model", df.index)
    return df.reset_index(drop=True).sort_values("macro_f1", ascending=False)


def selected_training_modes(training_mode: str, run_all_modes: bool) -> list[str]:
    return TABEP_TRAINING_MODES if run_all_modes else [training_mode]


def slugify_name(name: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-").replace("--", "-")


def resolve_uci_dataset(value: str) -> tuple[int, str]:
    normalized = value.strip().lower().replace("_", "-")
    if normalized.isdigit():
        dataset_id = int(normalized)
        return dataset_id, f"uci-{dataset_id}"
    if normalized not in UCI_DATASETS:
        choices = ", ".join(sorted(UCI_DATASETS))
        raise ValueError(f"Unknown UCI dataset {value!r}. Use an id or one of: {choices}")
    return UCI_DATASETS[normalized], normalized


def run_single_benchmark(
    bundle,
    *,
    dataset_label: str,
    args: argparse.Namespace,
    device: torch.device,
    progress: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    tabep_kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "free_steps": args.free_steps,
        "nudge_steps": args.nudge_steps,
        "dt": args.dt,
        "beta_nudge": args.beta_nudge,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "training_mode": args.training_mode,
        "guidance_weight": args.guidance_weight,
        "guidance_beta": args.guidance_beta,
        "hidden_state_l2": args.hidden_state_l2,
        "trajectory_guidance": not args.no_trajectory_guidance,
        "trajectory_consistency": args.trajectory_consistency,
        "trajectory_margin": args.trajectory_margin,
        "readout_type": args.readout_type,
        "prototypes_per_class": args.prototypes_per_class,
        "prototype_fusion": args.prototype_fusion,
        "interaction_rank": args.interaction_rank,
        "prototype_weight": args.prototype_weight,
        "readout_guidance": not args.no_readout_guidance,
        "calibrate_readout": args.calibrate_readout,
        "device": device,
        "tilelang_dynamics": args.tilelang_dynamics,
        "progress": progress,
    }
    training_modes = selected_training_modes(args.training_mode, args.all_training_modes)
    tabep_predictions: dict[str, np.ndarray] = {}
    if args.cv:
        x = np.concatenate([bundle.x_train, bundle.x_test], axis=0)
        y = np.concatenate([bundle.y_train, bundle.y_test], axis=0)
        results = run_baselines(x, y, seed=args.seed, cv_splits=args.cv_splits, device=device, progress=progress)
        if args.mlp_ablation:
            mlp_metrics, mlp_pred = train_mlp_cv(
                x,
                y,
                seed=args.seed,
                cv_splits=args.cv_splits,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                hidden_layers=args.hidden_layers,
                output_size=bundle.output_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                progress=progress,
            )
            results["MLP-matched"] = mlp_metrics
            tabep_predictions["MLP-matched"] = mlp_pred
        for mode in maybe_tqdm(training_modes, enabled=progress and len(training_modes) > 1, desc="TabEP modes", unit="mode"):
            mode_kwargs = {**tabep_kwargs, "training_mode": mode}
            tabep_metrics, tabep_pred = train_tabep_cv(
                x,
                y,
                seed=args.seed,
                cv_splits=args.cv_splits,
                **mode_kwargs,
            )
            model_name = f"TabEP-{mode}" if len(training_modes) > 1 else "TabEP"
            results[model_name] = tabep_metrics
            tabep_predictions[model_name] = tabep_pred
    else:
        x = np.concatenate([bundle.x_train, bundle.x_test], axis=0)
        results = run_baselines_holdout(bundle.x_train, bundle.x_test, bundle.y_train, bundle.y_test, seed=args.seed, device=device, progress=progress)
        if args.mlp_ablation:
            mlp_metrics, mlp_pred = train_mlp_holdout(
                bundle.x_train,
                bundle.x_test,
                bundle.y_train,
                bundle.y_test,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                hidden_layers=args.hidden_layers,
                output_size=bundle.output_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                progress=progress,
            )
            results["MLP-matched"] = mlp_metrics
            tabep_predictions["MLP-matched"] = mlp_pred
        for mode in maybe_tqdm(training_modes, enabled=progress and len(training_modes) > 1, desc="TabEP modes", unit="mode"):
            mode_kwargs = {**tabep_kwargs, "training_mode": mode}
            tabep_metrics, tabep_pred = train_tabep_holdout(
                bundle.x_train,
                bundle.x_test,
                bundle.y_train,
                bundle.y_test,
                seed=args.seed,
                **mode_kwargs,
            )
            model_name = f"TabEP-{mode}" if len(training_modes) > 1 else "TabEP"
            results[model_name] = tabep_metrics
            tabep_predictions[model_name] = tabep_pred
    df = dataframe_from_results(results)

    output_dir = args.output_dir / dataset_label if args.benchmark_suite else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{dataset_label}_benchmark.csv"
    json_path = output_dir / f"{dataset_label}_benchmark.json"
    df.to_csv(csv_path, index=False)
    for model_name, prediction in tabep_predictions.items():
        prediction_name = model_name.lower().replace("-", "_")
        np.save(output_dir / f"{prediction_name}_predictions.npy", prediction)
    json_path.write_text(json.dumps({k: {m: float(v) for m, v in vals.items()} for k, vals in results.items()}, indent=2), encoding="utf-8")

    print(f"{dataset_label} samples={x.shape[0]} features={x.shape[1]} classes={bundle.class_names}")
    print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return df, results


def main() -> None:
    install_tilelang_stderr_filter()
    parser = argparse.ArgumentParser(description="Drug200 TabEP benchmark")
    parser.add_argument("--dataset", default="milotix/drug200", help="Hugging Face dataset repo or local CSV path.")
    parser.add_argument("--uci-id", type=int, default=None, help="Load a UCI ML Repository dataset id through ucimlrepo.")
    parser.add_argument("--uci-name", default=None, help="Load a known UCI dataset by slug, e.g. iris, adult, shuttle.")
    parser.add_argument("--benchmark-suite", action="store_true", help="Run the requested UCI benchmark suite in order.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional stratified cap for large tabular datasets.")
    parser.add_argument("--split", default=None, help="Optional Hugging Face dataset split to load.")
    parser.add_argument("--csv", type=Path, default=None, help="Deprecated alias for --dataset with a local CSV path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--free-steps", type=int, default=2)
    parser.add_argument("--nudge-steps", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--beta-nudge", type=float, default=0.7)
    parser.add_argument("--training-mode", choices=["ep", "gd", "guided"], default="gd", help="TabEP objective: pure EP, pure gradient descent, or guided EP+GD.")
    parser.add_argument("--all-training-modes", action="store_true", help="Run TabEP with ep, gd, and guided objectives in one benchmark.")
    parser.add_argument("--mlp-ablation", action="store_true", help="Run a plain supervised MLP with approximately the same trainable parameter count as TabEP+readout.")
    parser.add_argument("--guidance-weight", type=float, default=1.0, help="Cross-entropy guidance weight for guided EP.")
    parser.add_argument("--guidance-beta", type=float, default=0.0, help="Optional target nudge inside the supervised guidance dynamics.")
    parser.add_argument("--hidden-state-l2", type=float, default=1e-4, help="State magnitude penalty for supervised/guided dynamics.")
    parser.add_argument("--no-trajectory-guidance", action="store_true", help="Disable trajectory supervision for GD TabEP.")
    parser.add_argument("--trajectory-consistency", type=float, default=0.1, help="KL consistency weight across relaxation-time logits for GD TabEP.")
    parser.add_argument("--trajectory-margin", type=float, default=0.05, help="Final-state classification margin penalty for GD TabEP.")
    parser.add_argument("--readout-type", choices=["mlp", "prototype", "hybrid-prototype"], default="mlp", help="TabEP readout head: MLP, pure prototype RBF, or hybrid linear+prototype RBF.")
    parser.add_argument("--prototypes-per-class", type=int, default=2, help="Number of learned RBF prototypes per class for prototype readouts.")
    parser.add_argument("--prototype-fusion", choices=["concat", "gate", "bilinear", "concat-bilinear"], default="concat", help="How prototype readouts fuse input features with the one-step energy state.")
    parser.add_argument("--interaction-rank", type=int, default=8, help="Low-rank m-by-n interaction rank for bilinear prototype fusion.")
    parser.add_argument("--prototype-weight", type=float, default=1e-3, help="Prototype separation regularization weight for prototype readouts.")
    parser.add_argument("--no-readout-guidance", action="store_true", help="Disable the trainable supervised readout used by GD/guided TabEP.")
    parser.add_argument("--calibrate-readout", action="store_true", help="Enable post-training logistic calibration for GD TabEP logits.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="Device for TabEP: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--tilelang-dynamics", action="store_true", help="Use TileLang kernels for TabEP inference dynamics on CUDA.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tabep-drug200"))
    parser.add_argument("--cv", action="store_true", help="Use slower stratified CV instead of the default holdout split.")
    args = parser.parse_args()

    set_seed(args.seed)
    progress = not args.no_progress
    device = resolve_device(args.device)
    if args.benchmark_suite:
        summary_rows = []
        for dataset_label, dataset_id in BENCHMARK_DATASETS:
            print(f"\n=== UCI {dataset_label} (id={dataset_id}) ===", flush=True)
            try:
                bundle, dataset_name = load_ucirepo_bundle(dataset_id, seed=args.seed, batch_size=args.batch_size, max_samples=args.max_samples)
                df, _ = run_single_benchmark(bundle, dataset_label=dataset_label, args=args, device=device, progress=progress)
                for row in df.to_dict(orient="records"):
                    summary_rows.append({"dataset": dataset_label, "uci_id": dataset_id, "uci_name": dataset_name, **row})
            except Exception as exc:
                print(f"Skipping {dataset_label} (id={dataset_id}): {exc}", flush=True)
                summary_rows.append({"dataset": dataset_label, "uci_id": dataset_id, "uci_name": "", "model": "ERROR", "error": str(exc)})
        summary_df = pd.DataFrame(summary_rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "uci_benchmark_summary.csv"
        summary_json_path = args.output_dir / "uci_benchmark_summary.json"
        summary_df.to_csv(summary_path, index=False)
        summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        print(f"\nWrote suite summary to {summary_path}")
        return

    if args.uci_id is not None or args.uci_name is not None:
        dataset_id, dataset_label = (args.uci_id, f"uci-{args.uci_id}") if args.uci_id is not None else resolve_uci_dataset(str(args.uci_name))
        bundle, dataset_name = load_ucirepo_bundle(dataset_id, seed=args.seed, batch_size=args.batch_size, max_samples=args.max_samples)
        run_single_benchmark(bundle, dataset_label=slugify_name(dataset_name) if args.uci_id is not None else dataset_label, args=args, device=device, progress=progress)
        return

    dataset_source = args.csv if args.csv is not None else args.dataset
    bundle = load_drug200_bundle(dataset_source, split=args.split, seed=args.seed, batch_size=args.batch_size, max_samples=args.max_samples)
    run_single_benchmark(bundle, dataset_label="drug200", args=args, device=device, progress=progress)


if __name__ == "__main__":
    main()
