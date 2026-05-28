from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from tabep.tabular import load_drug200_bundle, load_ucirepo_bundle
from tabep.tilelang_classifiers import TileLangKNeighborsClassifier, TileLangRbfSVC

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "tabep-drug200-proto" / "drug200_benchmark.csv"
UCI_RANKS = ROOT / "outputs" / "uci-suite-proto" / "uci_average_macro_f1_ranks_proto.csv"
UCI_SUMMARY = ROOT / "outputs" / "uci-suite-proto" / "uci_benchmark_summary_proto.csv"
SEED_SUMMARY = ROOT / "outputs" / "paper1-reproduction" / "seed_study_summary.csv"
OUT = ROOT / "papers" / "paper1" / "graphs"

PALETTE = {
    "TabEP": "#3B5BA9",
    "MLP-matched": "#5AA469",
    "LogisticRegression": "#D98C32",
    "TileLang-KNN": "#7B68A6",
    "TileLang-RBF-SVM": "#B64B4B",
    "DecisionTree": "#6B6B6B",
}

UCI_DATASETS = {
    "adult": 2,
    "breast-cancer-wisconsin-diagnostic": 17,
    "covertype": 31,
    "iris": 53,
    "wine": 109,
    "letter-recognition": 59,
    "optdigits": 80,
    "pendigits": 81,
    "satimage": 146,
    "segment": 50,
    "vehicle-silhouettes": 149,
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.0,
        edgecolor="#2B2B2B",
        facecolor=color,
        alpha=0.96,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9, color="#1F1F1F")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#2B2B2B", "shrinkA": 3, "shrinkB": 3},
    )


def architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    blue = "#DDE8F7"
    green = "#DDF3DE"
    red = "#F6CFCB"
    neutral = "#EFEFEF"
    yellow = "#FFF4C2"

    rounded_box(ax, (0.03, 0.56), 0.14, 0.18, "Tabular\nfeatures $x$", neutral)
    rounded_box(ax, (0.23, 0.56), 0.17, 0.18, "One-step\nenergy update", blue)
    rounded_box(ax, (0.46, 0.56), 0.16, 0.18, "Energy\nstate $h$", blue)
    rounded_box(ax, (0.46, 0.25), 0.16, 0.18, "Joint embedding\n$[x;h]$", yellow)
    rounded_box(ax, (0.68, 0.25), 0.15, 0.18, "Prototype/RBF\nclass scores", green)
    rounded_box(ax, (0.88, 0.25), 0.09, 0.18, r"Logits" "\n" r"$\ell$", red)

    arrow(ax, (0.17, 0.65), (0.23, 0.65))
    arrow(ax, (0.40, 0.65), (0.46, 0.65))
    arrow(ax, (0.54, 0.56), (0.54, 0.43))
    arrow(ax, (0.17, 0.60), (0.46, 0.34))
    arrow(ax, (0.62, 0.34), (0.68, 0.34))
    arrow(ax, (0.83, 0.34), (0.88, 0.34))

    ax.text(0.315, 0.83, "$h = s_1^{(L)}$", ha="center", va="center", fontsize=9, color="#3B5BA9")
    ax.text(0.755, 0.13, r"$\log\sum_k\exp(-\gamma\|z-p_{c,k}\|^2)$", ha="center", va="center", fontsize=9, color="#2F6F3E")
    ax.text(0.50, 0.04, "Selected TabEP: one relaxation step + direct joint embedding + hybrid prototype/RBF readout", ha="center", va="center", fontsize=9)
    save(fig, "tabep_architecture")


def metric_bars(df: pd.DataFrame) -> None:
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    labels = ["Accuracy", "Macro-F1", "Macro-Precision", "Macro-Recall"]
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2) * width
    colors = ["#3B5BA9", "#5AA469", "#D98C32", "#7B68A6"]
    for metric, label, offset, color in zip(metrics, labels, offsets, colors):
        values = df[metric].to_numpy()
        ax.bar(x + offset, values, width=width, label=label, color=color, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=28, ha="right")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    save(fig, "metric_comparison")


def macro_f1_rank(df: pd.DataFrame) -> None:
    order = df.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    bars = ax.barh(
        order["model"],
        order["macro_f1"],
        color=[PALETTE.get(model, "#777777") for model in order["model"]],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar in bars:
        value = bar.get_width()
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, min(1.0, max(order["macro_f1"]) + 0.16))
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    save(fig, "macro_f1_ranking")


def radar(df: pd.DataFrame) -> None:
    selected = df[df["model"].isin(["TabEP", "MLP-matched", "LogisticRegression", "TileLang-KNN"])]
    metrics = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    labels = ["Accuracy", "Macro-F1", "Macro-Precision", "Macro-Recall"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(4.5, 4.0))
    ax = fig.add_subplot(111, polar=True)
    for _, row in selected.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, label=row["model"], linewidth=1.8, color=PALETTE.get(row["model"], None))
        ax.fill(angles, values, alpha=0.08, color=PALETTE.get(row["model"], None))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 0.92)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["0.6", "0.7", "0.8", "0.9"])
    ax.grid(color="#DADADA", linewidth=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    save(fig, "radar_key_models")


def error_bars(summary: pd.DataFrame) -> None:
    order = summary.sort_values("macro_f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    colors = [PALETTE.get(model, "#777777") for model in order["model"]]
    ax.barh(order["model"], order["macro_f1_mean"], xerr=order["macro_f1_std"], color=colors, edgecolor="white", linewidth=0.5, capsize=3)
    for _, row in order.iterrows():
        ax.text(row["macro_f1_mean"] + row["macro_f1_std"] + 0.01, row["model"], f"{row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Macro-F1 across seeds")
    ax.set_xlim(0, min(1.08, float((order["macro_f1_mean"] + order["macro_f1_std"]).max()) + 0.18))
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    save(fig, "macro_f1_seed_error_bars")


def uci_average_rank(rank_df: pd.DataFrame) -> None:
    order = rank_df.sort_values("rank", ascending=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    colors = [PALETTE.get(model, "#777777") for model in order["model"]]
    bars = ax.barh(order["model"], order["rank"], color=colors, edgecolor="black", linewidth=0.6)
    ax.invert_yaxis()
    for bar in bars:
        value = bar.get_width()
        ax.text(value + 0.04, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_xlim(0, max(5.0, float(order["rank"].max()) + 0.4))
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    save(fig, "uci_average_rank")


def uci_tabep_gap(summary: pd.DataFrame) -> None:
    data = summary.copy()
    data["gap"] = data["tab_f1"] - data["best_f1"]
    data = data.sort_values("gap", ascending=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colors = np.where(data["gap"] >= -1e-12, "#5AA469", "#B64B4B")
    bars = ax.barh(data["dataset"], data["gap"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="#272727", linewidth=1.0)
    for bar in bars:
        value = bar.get_width()
        x = value - 0.006 if value < 0 else value + 0.006
        ha = "right" if value < 0 else "left"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", ha=ha, fontsize=7)
    ax.set_xlabel("TabEP Macro-F1 minus best Macro-F1")
    ax.set_xlim(min(-0.16, float(data["gap"].min()) - 0.03), 0.04)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    save(fig, "uci_tabep_gap")


def confusion_matrix_grid() -> None:
    bundle = load_drug200_bundle(
        ROOT / "data" / "raw" / "drug200" / "drug200.csv",
        test_size=0.8,
        seed=42,
        batch_size=32,
        eval_batch_size=256,
        num_workers=0,
    )
    predictions: dict[str, np.ndarray] = {}
    prediction_files = [
        ("TabEP", ROOT / "outputs" / "tabep-drug200-proto" / "tabep_predictions.npy"),
        ("MLP-matched", ROOT / "outputs" / "tabep-drug200-proto" / "mlp_matched_predictions.npy"),
    ]
    for name, path in prediction_files:
        if path.exists():
            predictions[name] = np.load(path)

    baseline_models = {
        "LogisticRegression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
        "TileLang-KNN": TileLangKNeighborsClassifier(n_neighbors=5, weights="distance", device=torch.device("cpu")),
        "TileLang-RBF-SVM": TileLangRbfSVC(C=3.0, gamma="scale", class_weight="balanced", device=torch.device("cpu")),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
    }
    for name, model in baseline_models.items():
        model.fit(bundle.x_train, bundle.y_train)
        predictions[name] = model.predict(bundle.x_test)

    ordered_names = ["TabEP", "MLP-matched", "LogisticRegression", "TileLang-KNN", "TileLang-RBF-SVM", "DecisionTree"]
    ordered_names = [name for name in ordered_names if name in predictions]
    if not ordered_names:
        return

    labels = list(range(len(bundle.class_names)))
    ncols = 3
    nrows = int(np.ceil(len(ordered_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 4.8), squeeze=False)
    last_image = None
    for ax, name in zip(axes.ravel(), ordered_names):
        pred = predictions[name]
        cm = confusion_matrix(bundle.y_test, pred, labels=labels, normalize="true")
        last_image = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=9)
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(bundle.class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(bundle.class_names, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        for row in labels:
            for col in labels:
                value = cm[row, col]
                color = "white" if value > 0.55 else "#1F1F1F"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.bottom.set_visible(False)
    for ax in axes.ravel()[len(ordered_names):]:
        ax.set_axis_off()
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label("Row-normalized recall")
    fig.subplots_adjust(wspace=0.34, hspace=0.72, right=0.88)
    fig.savefig(OUT / "drug200_confusion_matrices.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "drug200_confusion_matrices.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def tabep_uci_confusion_grid() -> None:
    summary_path = ROOT / "outputs" / "uci-suite-proto" / "uci_benchmark_summary_proto.csv"
    pred_root = ROOT / "outputs" / "uci-suite-proto-predictions"
    if not summary_path.exists() or not pred_root.exists():
        return

    summary = pd.read_csv(summary_path)
    priority = summary.sort_values("tab_f1", ascending=False)["dataset"].tolist()
    datasets: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    for dataset in priority:
        if dataset not in UCI_DATASETS:
            continue
        pred_path = pred_root / dataset / "tabep_predictions.npy"
        if not pred_path.exists():
            continue
        try:
            bundle, _ = load_ucirepo_bundle(
                UCI_DATASETS[dataset],
                seed=42,
                batch_size=256,
                eval_batch_size=256,
                num_workers=0,
                max_samples=5000,
            )
        except Exception:
            continue
        pred = np.load(pred_path)
        if pred.shape[0] != bundle.y_test.shape[0] or bundle.output_size > 10:
            continue
        datasets.append((dataset, bundle.y_test, pred, bundle.output_size))
        if len(datasets) == 9:
            break

    if not datasets:
        return

    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.35 * nrows), squeeze=False)
    last_image = None
    for ax, (dataset, y_true, pred, num_classes) in zip(axes.ravel(), datasets):
        labels = list(range(num_classes))
        cm = confusion_matrix(y_true, pred, labels=labels, normalize="true")
        last_image = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
        pretty = dataset.replace("breast-cancer-wisconsin-diagnostic", "breast cancer").replace("vehicle-silhouettes", "vehicle").replace("letter-recognition", "letter")
        ax.set_title(pretty, fontsize=8)
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels([str(label) for label in labels], fontsize=6)
        ax.set_yticklabels([str(label) for label in labels], fontsize=6)
        ax.set_xlabel("Pred.", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
        if num_classes <= 7:
            for row in labels:
                for col in labels:
                    value = cm[row, col]
                    color = "white" if value > 0.55 else "#1F1F1F"
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.6, color=color)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(datasets):]:
        ax.set_axis_off()
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("Row-normalized recall")
    fig.subplots_adjust(wspace=0.42, hspace=0.58, right=0.88)
    fig.savefig(OUT / "tabep_uci_confusion_grid.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "tabep_uci_confusion_grid.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(RESULTS)
    architecture_diagram()
    metric_bars(df)
    macro_f1_rank(df)
    radar(df)
    if SEED_SUMMARY.exists():
        error_bars(pd.read_csv(SEED_SUMMARY))
    if UCI_RANKS.exists():
        uci_average_rank(pd.read_csv(UCI_RANKS))
    if UCI_SUMMARY.exists():
        uci_tabep_gap(pd.read_csv(UCI_SUMMARY))
    confusion_matrix_grid()
    tabep_uci_confusion_grid()


if __name__ == "__main__":
    main()
