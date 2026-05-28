from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from tabep.tabular import load_drug200_bundle
from tabep.tabular_benchmark import train_mlp_holdout, train_tabep_holdout
from tabep.tilelang_classifiers import TileLangKNeighborsClassifier, TileLangRbfSVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper1-reproduction"
SEEDS = [42, 43, 44]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        bundle = load_drug200_bundle("milotix/drug200", seed=seed, batch_size=32)
        x_train, x_test, y_train, y_test = bundle.x_train, bundle.x_test, bundle.y_train, bundle.y_test

        classical = {
            "TileLang-KNN": TileLangKNeighborsClassifier(n_neighbors=5, weights="distance", device=device),
            "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=seed),
            "TileLang-RBF-SVM": TileLangRbfSVC(C=3.0, gamma="scale", class_weight="balanced", device=device),
            "LogisticRegression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
        }
        for name, model in classical.items():
            model.fit(x_train, y_train)
            row = {"seed": seed, "model": name}
            row.update(metrics(y_test, model.predict(x_test)))
            rows.append(row)

        mlp_metrics, _ = train_mlp_holdout(
            x_train,
            x_test,
            y_train,
            y_test,
            seed=seed,
            epochs=100,
            batch_size=32,
            hidden_size=48,
            hidden_layers=2,
            output_size=bundle.output_size,
            lr=1e-2,
            weight_decay=1e-4,
            device=device,
            progress=False,
        )
        rows.append({"seed": seed, "model": "MLP-matched", **mlp_metrics})

        tabep_metrics, _ = train_tabep_holdout(
            x_train,
            x_test,
            y_train,
            y_test,
            seed=seed,
            epochs=100,
            batch_size=32,
            hidden_size=48,
            hidden_layers=2,
            free_steps=2,
            nudge_steps=6,
            dt=0.08,
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
        rows.append({"seed": seed, "model": "TabEP", **tabep_metrics})

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("model")[["accuracy", "macro_f1", "macro_precision", "macro_recall"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["model"] + [f"{metric}_{stat}" for metric, stat in summary.columns[1:]]
    summary = summary.sort_values("macro_f1_mean", ascending=False)

    raw.to_csv(OUT / "seed_study_raw.csv", index=False)
    summary.to_csv(OUT / "seed_study_summary.csv", index=False)
    (OUT / "seed_study_summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8")
    print(raw.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSummary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved: {OUT / 'seed_study_summary.csv'}")


if __name__ == "__main__":
    main()
