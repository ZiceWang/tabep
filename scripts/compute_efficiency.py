from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
from fvcore.nn import FlopCountAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from tabep.model import DeepEnergyModel
from tabep.tabular import load_drug200_bundle
from tabep.tabular_benchmark import MLPClassifier, TabEPReadout, matched_mlp_width, make_readout, readout_logits, train_mlp_holdout, train_tabep_holdout
from tabep.tilelang_classifiers import TileLangKNeighborsClassifier, TileLangRbfSVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper1-reproduction"
CSV = OUT / "efficiency_results.csv"
JSON = OUT / "efficiency_results.json"

SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_SIZE = 48
HIDDEN_LAYERS = 2
FREE_STEPS = 2
DT = 0.08


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, *, repeats: int = 50, warmup: int = 10, device: torch.device) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(samples)), float(median(samples))


def sklearn_time(model, x_train, y_train, x_test) -> tuple[float, float]:
    model.fit(x_train, y_train)
    return time_call(lambda: model.predict(x_test), repeats=200, warmup=20, device=torch.device("cpu"))


def estimate_tabep_flops(model: DeepEnergyModel, readout: TabEPReadout, x_sample: torch.Tensor) -> int:
    class Wrapper(torch.nn.Module):
        def __init__(self, model: DeepEnergyModel, readout: TabEPReadout) -> None:
            super().__init__()
            self.model = model
            self.readout = readout

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            states = self.model.run_dynamics(x, steps=FREE_STEPS, dt=DT)
            return readout_logits(self.readout, states[-1], x)

    wrapper = Wrapper(model, readout).eval()
    return int(FlopCountAnalysis(wrapper, x_sample).total() / x_sample.shape[0])


def estimate_mlp_flops(model: MLPClassifier, x_sample: torch.Tensor) -> int:
    return int(FlopCountAnalysis(model.eval(), x_sample).total() / x_sample.shape[0])


def formula_flops(model_name: str, *, n_train: int, n_test: int, n_features: int, n_classes: int, n_nodes: int | None = None) -> tuple[int, str]:
    if model_name == "TileLang-KNN":
        per_sample = n_train * (3 * n_features) + n_train
        return int(per_sample), "Analytic: pairwise squared distances plus voting"
    if model_name == "TileLang-RBF-SVM":
        per_sample = n_train * (3 * n_features + 1) + n_train * n_classes
        return int(per_sample), "Analytic: RBF distances plus class score matmul"
    if model_name == "LogisticRegression":
        return int(2 * n_features * n_classes), "Analytic: dense linear classifier"
    if model_name == "DecisionTree":
        depth = int(np.ceil(np.log2(max(n_nodes or 2, 2))))
        return depth, "Analytic proxy: one feature comparison per tree level"
    raise ValueError(model_name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_drug200_bundle("milotix/drug200", seed=SEED, batch_size=BATCH_SIZE)
    x_train, x_test, y_train, y_test = bundle.x_train, bundle.x_test, bundle.y_train, bundle.y_test
    n_train, n_test, n_features, n_classes = x_train.shape[0], x_test.shape[0], x_train.shape[1], bundle.output_size

    rows: list[dict[str, object]] = []

    # Classical models: library/runtime timing plus analytic operation estimates.
    classical = [
        ("TileLang-KNN", TileLangKNeighborsClassifier(n_neighbors=5, weights="distance", device=device)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=4, random_state=SEED)),
        ("TileLang-RBF-SVM", TileLangRbfSVC(C=3.0, gamma="scale", class_weight="balanced", device=device)),
        ("LogisticRegression", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)),
    ]
    for name, model in classical:
        mean_ms, median_ms = sklearn_time(model, x_train, y_train, x_test)
        n_nodes = getattr(model, "tree_", None).node_count if name == "DecisionTree" else None
        flops, source = formula_flops(name, n_train=n_train, n_test=n_test, n_features=n_features, n_classes=n_classes, n_nodes=n_nodes)
        rows.append(
            {
                "model": name,
                "flops_per_sample": flops,
                "flops_source": source,
                "latency_ms_mean": mean_ms,
                "latency_ms_median": median_ms,
                "throughput_samples_per_s": 1000.0 * n_test / mean_ms,
                "device": str(device),
            }
        )

    # TabEP and MLP are trained with the benchmark helpers, then reconstructed for fvcore FLOPs.
    train_tabep_holdout(
        x_train,
        x_test,
        y_train,
        y_test,
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        free_steps=FREE_STEPS,
        nudge_steps=6,
        dt=DT,
        beta_nudge=0.7,
        lr=1e-2,
        weight_decay=1e-4,
        training_mode="gd",
        guidance_weight=1.0,
        guidance_beta=0.0,
        hidden_state_l2=1e-4,
        readout_guidance=True,
        calibrate_readout=False,
        trajectory_guidance=True,
        trajectory_consistency=0.1,
        trajectory_margin=0.05,
        device=device,
        tilelang_dynamics=False,
        progress=False,
    )
    # Refit lightweight copies for timing/FLOP measurement to keep this script self-contained.
    torch.manual_seed(SEED)
    tabep = DeepEnergyModel([n_features, HIDDEN_SIZE, HIDDEN_SIZE, n_classes], rho="hardtanh", weight_scale=0.035, fhn_delta=0.75, fhn_epsilon=0.35, fhn_alpha=0.75).to(device)
    tabep_readout = make_readout(n_features, n_classes, n_classes, HIDDEN_SIZE, device)
    tabep_params = sum(p.numel() for p in list(tabep.parameters()) + list(tabep_readout.parameters()))
    x_tensor = torch.from_numpy(x_test.astype(np.float32)).to(device)
    tabep_mean, tabep_median = time_call(lambda: readout_logits(tabep_readout, tabep.predict(x_tensor, steps=FREE_STEPS, dt=DT), x_tensor).argmax(dim=1), repeats=100, warmup=20, device=device)
    rows.append(
        {
            "model": "TabEP",
            "flops_per_sample": estimate_tabep_flops(tabep, tabep_readout, x_tensor[:1]),
            "flops_source": "fvcore FlopCountAnalysis on one-sample forward",
            "latency_ms_mean": tabep_mean,
            "latency_ms_median": tabep_median,
            "throughput_samples_per_s": 1000.0 * n_test / tabep_mean,
            "device": str(device),
            "parameters": tabep_params,
        }
    )

    train_mlp_holdout(
        x_train,
        x_test,
        y_train,
        y_test,
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        hidden_layers=HIDDEN_LAYERS,
        output_size=n_classes,
        lr=1e-2,
        weight_decay=1e-4,
        device=device,
        progress=False,
    )
    target_params = tabep_params
    mlp_width = matched_mlp_width(n_features, n_classes, HIDDEN_LAYERS, target_params)
    mlp = MLPClassifier(n_features, mlp_width, n_classes, HIDDEN_LAYERS).to(device)
    mlp_params = sum(p.numel() for p in mlp.parameters())
    mlp_mean, mlp_median = time_call(lambda: mlp(x_tensor).argmax(dim=1), repeats=200, warmup=20, device=device)
    rows.append(
        {
            "model": "MLP-matched",
            "flops_per_sample": estimate_mlp_flops(mlp, x_tensor[:1]),
            "flops_source": "fvcore FlopCountAnalysis on one-sample forward",
            "latency_ms_mean": mlp_mean,
            "latency_ms_median": mlp_median,
            "throughput_samples_per_s": 1000.0 * n_test / mlp_mean,
            "device": str(device),
            "parameters": mlp_params,
        }
    )

    df = pd.DataFrame(rows)
    df = df[["model", "flops_per_sample", "latency_ms_mean", "latency_ms_median", "throughput_samples_per_s", "device", "parameters", "flops_source"]]
    df.to_csv(CSV, index=False)
    JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {CSV}")


if __name__ == "__main__":
    main()
