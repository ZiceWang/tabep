from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from .tilelang_utils import install_tilelang_stderr_filter, silence_stderr_fd

install_tilelang_stderr_filter()

try:
    with silence_stderr_fd():
        import tilelang
        import tilelang.language as T
except Exception:  # pragma: no cover - TileLang is optional at import time.
    tilelang = None
    T = None


@lru_cache(maxsize=32)
def _pairwise_sqdist_kernel(m: int, n: int, d: int, block_m: int = 16, block_n: int = 16):
    if tilelang is None or T is None:
        raise RuntimeError("TileLang is not available.")

    @tilelang.jit(target="cuda")
    def pairwise_sqdist():
        @T.prim_func
        def kernel(
            X: T.Tensor((m, d), "float32"),  # type: ignore[valid-type]
            Y: T.Tensor((n, d), "float32"),  # type: ignore[valid-type]
            Out: T.Tensor((m, n), "float32"),  # type: ignore[valid-type]
        ):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=(block_m, block_n)) as (bx, by):
                for ti, tj in T.Parallel(block_m, block_n):
                    row = by * block_m + ti
                    col = bx * block_n + tj
                    acc = T.alloc_local((), "float32")
                    acc[()] = 0.0
                    for k in T.serial(d):
                        diff = X[row, k] - Y[col, k]
                        acc[()] += diff * diff
                    Out[row, col] = acc[()]

        return kernel

    return pairwise_sqdist()


@lru_cache(maxsize=32)
def _rbf_kernel(m: int, n: int, d: int, gamma: float, block_m: int = 16, block_n: int = 16):
    if tilelang is None or T is None:
        raise RuntimeError("TileLang is not available.")

    @tilelang.jit(target="cuda")
    def rbf():
        @T.prim_func
        def kernel(
            X: T.Tensor((m, d), "float32"),  # type: ignore[valid-type]
            Y: T.Tensor((n, d), "float32"),  # type: ignore[valid-type]
            Out: T.Tensor((m, n), "float32"),  # type: ignore[valid-type]
        ):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=(block_m, block_n)) as (bx, by):
                for ti, tj in T.Parallel(block_m, block_n):
                    row = by * block_m + ti
                    col = bx * block_n + tj
                    acc = T.alloc_local((), "float32")
                    acc[()] = 0.0
                    for k in T.serial(d):
                        diff = X[row, k] - Y[col, k]
                        acc[()] += diff * diff
                    Out[row, col] = T.exp(-gamma * acc[()])

        return kernel

    return rbf()


def _as_float_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=device)


def _pairwise_sqdist(x: torch.Tensor, y: torch.Tensor, use_tilelang: bool) -> torch.Tensor:
    if use_tilelang and x.is_cuda and y.is_cuda:
        out = torch.empty((x.shape[0], y.shape[0]), device=x.device, dtype=torch.float32)
        _pairwise_sqdist_kernel(x.shape[0], y.shape[0], x.shape[1])(x, y, out)
        return out
    return torch.cdist(x, y).square()


def _pairwise_rbf(x: torch.Tensor, y: torch.Tensor, gamma: float, use_tilelang: bool) -> torch.Tensor:
    if use_tilelang and x.is_cuda and y.is_cuda:
        out = torch.empty((x.shape[0], y.shape[0]), device=x.device, dtype=torch.float32)
        _rbf_kernel(x.shape[0], y.shape[0], x.shape[1], float(gamma))(x, y, out)
        return out
    return torch.exp(-gamma * torch.cdist(x, y).square())


def _resolve_gamma(gamma: str | float, x: np.ndarray) -> float:
    if isinstance(gamma, str):
        if gamma == "scale":
            variance = float(np.var(x))
            return 1.0 / (x.shape[1] * variance) if variance > 0.0 else 1.0
        if gamma == "auto":
            return 1.0 / x.shape[1]
        raise ValueError(f"Unsupported gamma={gamma!r}.")
    return float(gamma)


@dataclass
class TileLangKNeighborsClassifier:
    n_neighbors: int = 5
    weights: str = "distance"
    device: torch.device | str = "auto"
    use_tilelang: bool = True
    chunk_size: int = 4096

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "device": self.device,
            "use_tilelang": self.use_tilelang,
            "chunk_size": self.chunk_size,
        }

    def set_params(self, **params: Any) -> TileLangKNeighborsClassifier:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> TileLangKNeighborsClassifier:
        device = _classifier_device(self.device)
        self.x_train_ = _as_float_tensor(x, device)
        self.y_train_ = torch.as_tensor(np.asarray(y, dtype=np.int64), device=device)
        self.classes_ = torch.unique(self.y_train_, sorted=True)
        self.n_classes_ = int(self.classes_.numel())
        self.n_neighbors_ = min(self.n_neighbors, int(self.x_train_.shape[0]))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_query = _as_float_tensor(x, self.x_train_.device)
        outputs: list[torch.Tensor] = []
        for start in range(0, x_query.shape[0], self.chunk_size):
            query_chunk = x_query[start : start + self.chunk_size]
            dist2 = _pairwise_sqdist(query_chunk, self.x_train_, self.use_tilelang)
            values, indices = torch.topk(dist2, k=self.n_neighbors_, largest=False, dim=1)
            labels = self.y_train_[indices]
            if self.weights == "distance":
                weights = 1.0 / (torch.sqrt(torch.clamp(values, min=0.0)) + 1e-12)
            elif self.weights == "uniform":
                weights = torch.ones_like(values)
            else:
                raise ValueError(f"Unsupported weights={self.weights!r}.")
            scores = torch.zeros((query_chunk.shape[0], self.n_classes_), device=query_chunk.device, dtype=torch.float32)
            scores.scatter_add_(1, labels, weights)
            outputs.append(self.classes_[scores.argmax(dim=1)])
        return torch.cat(outputs).detach().cpu().numpy()


@dataclass
class TileLangRbfSVC:
    C: float = 3.0
    gamma: str | float = "scale"
    class_weight: str | None = "balanced"
    device: torch.device | str = "auto"
    use_tilelang: bool = True
    chunk_size: int = 4096

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "C": self.C,
            "gamma": self.gamma,
            "class_weight": self.class_weight,
            "device": self.device,
            "use_tilelang": self.use_tilelang,
            "chunk_size": self.chunk_size,
        }

    def set_params(self, **params: Any) -> TileLangRbfSVC:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, x: np.ndarray, y: np.ndarray) -> TileLangRbfSVC:
        device = _classifier_device(self.device)
        x_np = np.ascontiguousarray(x, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        self.x_train_ = _as_float_tensor(x_np, device)
        self.y_train_ = torch.as_tensor(y_np, device=device)
        self.classes_ = torch.unique(self.y_train_, sorted=True)
        self.n_classes_ = int(self.classes_.numel())
        self.gamma_ = _resolve_gamma(self.gamma, x_np)

        targets = torch.full((x_np.shape[0], self.n_classes_), -1.0, device=device, dtype=torch.float32)
        targets.scatter_(1, self.y_train_.unsqueeze(1), 1.0)
        if self.class_weight == "balanced":
            counts = torch.bincount(self.y_train_, minlength=self.n_classes_).float().clamp_min(1.0)
            sample_weights = x_np.shape[0] / (self.n_classes_ * counts[self.y_train_])
            targets = targets * sample_weights.unsqueeze(1)
        elif self.class_weight is not None:
            raise ValueError(f"Unsupported class_weight={self.class_weight!r}.")

        gram = _pairwise_rbf(self.x_train_, self.x_train_, self.gamma_, self.use_tilelang)
        regularizer = 1.0 / max(float(self.C), 1e-12)
        gram = gram + regularizer * torch.eye(gram.shape[0], device=device, dtype=torch.float32)
        self.alpha_ = torch.linalg.solve(gram, targets)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_query = _as_float_tensor(x, self.x_train_.device)
        outputs: list[torch.Tensor] = []
        for start in range(0, x_query.shape[0], self.chunk_size):
            query_chunk = x_query[start : start + self.chunk_size]
            kernel = _pairwise_rbf(query_chunk, self.x_train_, self.gamma_, self.use_tilelang)
            scores = kernel @ self.alpha_
            outputs.append(self.classes_[scores.argmax(dim=1)])
        return torch.cat(outputs).detach().cpu().numpy()


def _classifier_device(device: torch.device | str) -> torch.device:
    requested = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return requested
