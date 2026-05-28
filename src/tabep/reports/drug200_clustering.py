from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabep.tabular import load_drug200_dataframe

FEATURES = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
CATEGORICAL_ORDERS = {
    "Sex": ["F", "M"],
    "BP": ["LOW", "NORMAL", "HIGH"],
    "Cholesterol": ["NORMAL", "HIGH"],
}
DRUG_ORDER = ["drugA", "drugB", "drugC", "drugX", "DrugY"]
METHOD_ORDER = ["kmeans", "fuzzy_cmeans", "dbscan", "agglomerative", "nsga2"]
METHOD_NAMES = {
    "kmeans": "K-means",
    "fuzzy_cmeans": "Fuzzy C-means",
    "dbscan": "DBSCAN",
    "agglomerative": "Agglomerative",
    "nsga2": "NSGA-II multi-objective",
}
PUBLICATION_RCPARAMS = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 15,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2.0,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
}
METHOD_COLORS = {
    "kmeans": PALETTE["blue_secondary"],
    "fuzzy_cmeans": PALETTE["green_3"],
    "dbscan": PALETTE["red_2"],
    "agglomerative": PALETTE["blue_main"],
    "nsga2": PALETTE["highlight"],
}
CLUSTER_COLORS = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_strong"], "#42949E", "#9A4D8E", PALETTE["neutral_dark"]]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


plt.rcParams.update(PUBLICATION_RCPARAMS)


def encode_feature(series: pd.Series, name: str) -> np.ndarray:
    if name in CATEGORICAL_ORDERS:
        mapping = {value: idx for idx, value in enumerate(CATEGORICAL_ORDERS[name])}
        return series.map(mapping).astype(float).to_numpy()
    return series.astype(float).to_numpy()


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std, mean, std


def squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    labels = labels.copy()
    valid = sorted([lab for lab in set(labels.tolist()) if lab >= 0], key=lambda v: (-np.sum(labels == v), v))
    mapping = {old: new for new, old in enumerate(valid)}
    return np.array([mapping.get(int(lab), -1) for lab in labels], dtype=int)


def kmeans(x: np.ndarray, k: int, *, seed: int = 42, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = [x[int(rng.integers(len(x)))]]
    while len(centers) < k:
        d2 = np.min(squared_distances(x, np.vstack(centers)), axis=1)
        total = float(d2.sum())
        if total <= 1e-12:
            idx = int(rng.integers(len(x)))
        else:
            idx = int(np.searchsorted(np.cumsum(d2 / total), rng.random()))
        centers.append(x[idx])
    centers = np.vstack(centers).astype(float)

    labels = np.zeros(len(x), dtype=int)
    for _ in range(max_iter):
        new_labels = np.argmin(squared_distances(x, centers), axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = new_labels == j
            if np.any(mask):
                new_centers[j] = x[mask].mean(axis=0)
            else:
                new_centers[j] = x[int(rng.integers(len(x)))]
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break
        labels, centers = new_labels, new_centers
    return relabel_by_size(labels), centers


def fuzzy_cmeans(x: np.ndarray, k: int, *, seed: int = 42, m: float = 2.0, max_iter: int = 120, tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.random((len(x), k))
    u = u / u.sum(axis=1, keepdims=True)
    centers = np.zeros((k, x.shape[1]), dtype=float)
    for _ in range(max_iter):
        old_u = u.copy()
        um = u**m
        centers = (um.T @ x) / np.maximum(um.sum(axis=0)[:, None], 1e-12)
        dist = np.sqrt(np.maximum(squared_distances(x, centers), 1e-12))
        power = 2.0 / (m - 1.0)
        for i in range(len(x)):
            for j in range(k):
                ratio_sum = np.sum((dist[i, j] / dist[i]) ** power)
                u[i, j] = 1.0 / max(ratio_sum, 1e-12)
        if np.max(np.abs(u - old_u)) < tol:
            break
    labels = relabel_by_size(np.argmax(u, axis=1))
    return labels, centers, u


def dbscan(x: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = len(x)
    labels = np.full(n, -99, dtype=int)
    dists = np.sqrt(np.maximum(squared_distances(x, x), 0.0))
    neighbors = [np.where(dists[i] <= eps)[0].tolist() for i in range(n)]
    cluster_id = 0
    for i in range(n):
        if labels[i] != -99:
            continue
        if len(neighbors[i]) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = list(neighbors[i])
        pos = 0
        while pos < len(seeds):
            q = seeds[pos]
            if labels[q] == -1:
                labels[q] = cluster_id
            if labels[q] != -99:
                pos += 1
                continue
            labels[q] = cluster_id
            if len(neighbors[q]) >= min_samples:
                for nb in neighbors[q]:
                    if nb not in seeds:
                        seeds.append(nb)
            pos += 1
        cluster_id += 1
    labels[labels == -99] = -1
    return relabel_by_size(labels)


def choose_dbscan_params(x: np.ndarray) -> tuple[float, int]:
    min_samples = 5
    dists = np.sqrt(np.maximum(squared_distances(x, x), 0.0))
    sorted_d = np.sort(dists, axis=1)
    kth = sorted_d[:, min(min_samples, len(x) - 1)]
    best_eps = float(np.percentile(kth, 60))
    best_score = -1e18
    for q in range(45, 86, 5):
        eps = float(np.percentile(kth, q))
        labels = dbscan(x, eps, min_samples)
        k = cluster_count(labels)
        noise = float(np.mean(labels < 0))
        if 2 <= k <= 6:
            score = silhouette_score(x, labels) - 0.25 * noise
        else:
            score = -10.0 - abs(k - 3) - noise
        if score > best_score:
            best_score = score
            best_eps = eps
    return best_eps, min_samples


def agglomerative(x: np.ndarray, k: int) -> np.ndarray:
    clusters: list[list[int]] = [[i] for i in range(len(x))]
    centroids: list[np.ndarray] = [x[i].copy() for i in range(len(x))]
    sizes = [1 for _ in range(len(x))]
    while len(clusters) > k:
        best_pair = (0, 1)
        best_cost = math.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cost = sizes[i] * sizes[j] / (sizes[i] + sizes[j]) * np.sum((centroids[i] - centroids[j]) ** 2)
                if cost < best_cost:
                    best_cost = float(cost)
                    best_pair = (i, j)
        i, j = best_pair
        clusters[i].extend(clusters[j])
        new_size = sizes[i] + sizes[j]
        centroids[i] = (sizes[i] * centroids[i] + sizes[j] * centroids[j]) / new_size
        sizes[i] = new_size
        del clusters[j], centroids[j], sizes[j]
    labels = np.zeros(len(x), dtype=int)
    for cid, members in enumerate(clusters):
        labels[members] = cid
    return relabel_by_size(labels)


def cluster_count(labels: np.ndarray) -> int:
    return len({int(v) for v in labels if v >= 0})


def compactness(x: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        pts = x[labels == lab]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        total += float(np.sum((pts - center) ** 2))
    return total / max(np.sum(labels >= 0), 1)


def separation(x: np.ndarray, labels: np.ndarray) -> float:
    centers = []
    for lab in sorted(set(labels.tolist())):
        if lab >= 0:
            pts = x[labels == lab]
            if len(pts) > 0:
                centers.append(pts.mean(axis=0))
    if len(centers) < 2:
        return 0.0
    vals = [euclidean(centers[i], centers[j]) for i in range(len(centers)) for j in range(i + 1, len(centers))]
    return float(min(vals))


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    valid = labels >= 0
    labs = labels[valid]
    pts = x[valid]
    unique = sorted(set(labs.tolist()))
    if len(unique) < 2 or len(pts) < 3:
        return 0.0
    d = np.sqrt(np.maximum(squared_distances(pts, pts), 0.0))
    scores = []
    for i in range(len(pts)):
        same = labs == labs[i]
        if np.sum(same) <= 1:
            a = 0.0
        else:
            a = float(np.sum(d[i, same]) / (np.sum(same) - 1))
        b = math.inf
        for lab in unique:
            if lab == labs[i]:
                continue
            other = labs == lab
            b = min(b, float(np.mean(d[i, other])))
        denom = max(a, b)
        scores.append(0.0 if denom <= 1e-12 else (b - a) / denom)
    return float(np.mean(scores))


def davies_bouldin_index(x: np.ndarray, labels: np.ndarray) -> float:
    centers = []
    scatters = []
    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        pts = x[labels == lab]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        centers.append(center)
        scatters.append(float(np.mean(np.sqrt(np.sum((pts - center) ** 2, axis=1)))))
    if len(centers) < 2:
        return float("nan")
    vals = []
    for i in range(len(centers)):
        worst = -math.inf
        for j in range(len(centers)):
            if i == j:
                continue
            dist = euclidean(centers[i], centers[j])
            worst = max(worst, (scatters[i] + scatters[j]) / max(dist, 1e-12))
        vals.append(worst)
    return float(np.mean(vals))


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    mask = labels_pred >= 0
    labels_true = labels_true[mask]
    labels_pred = labels_pred[mask]
    n = len(labels_true)
    if n < 2:
        return 0.0

    def comb2(v: int) -> float:
        return v * (v - 1) / 2.0

    contingency: dict[tuple[int, int], int] = defaultdict(int)
    true_count: Counter[int] = Counter()
    pred_count: Counter[int] = Counter()
    for t, p in zip(labels_true.tolist(), labels_pred.tolist(), strict=False):
        contingency[(int(t), int(p))] += 1
        true_count[int(t)] += 1
        pred_count[int(p)] += 1
    sum_comb = sum(comb2(v) for v in contingency.values())
    sum_true = sum(comb2(v) for v in true_count.values())
    sum_pred = sum(comb2(v) for v in pred_count.values())
    total = comb2(n)
    expected = sum_true * sum_pred / total if total > 0 else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    return 0.0 if abs(denom) < 1e-12 else float((sum_comb - expected) / denom)


def normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    mask = labels_pred >= 0
    labels_true = labels_true[mask]
    labels_pred = labels_pred[mask]
    n = len(labels_true)
    if n == 0:
        return 0.0
    true_vals = sorted(set(labels_true.tolist()))
    pred_vals = sorted(set(labels_pred.tolist()))
    true_count = {t: int(np.sum(labels_true == t)) for t in true_vals}
    pred_count = {p: int(np.sum(labels_pred == p)) for p in pred_vals}
    mi = 0.0
    for t in true_vals:
        for p in pred_vals:
            nij = int(np.sum((labels_true == t) & (labels_pred == p)))
            if nij > 0:
                mi += (nij / n) * math.log((nij * n) / (true_count[t] * pred_count[p]))
    ht = -sum((c / n) * math.log(c / n) for c in true_count.values() if c > 0)
    hp = -sum((c / n) * math.log(c / n) for c in pred_count.values() if c > 0)
    denom = math.sqrt(ht * hp)
    return 0.0 if denom <= 1e-12 else float(mi / denom)


def purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    mask = labels_pred >= 0
    labels_true = labels_true[mask]
    labels_pred = labels_pred[mask]
    if len(labels_true) == 0:
        return 0.0
    total = 0
    for lab in sorted(set(labels_pred.tolist())):
        cnt = Counter(labels_true[labels_pred == lab].tolist())
        total += max(cnt.values()) if cnt else 0
    return float(total / len(labels_true))


@dataclass
class Individual:
    k: int
    centers: np.ndarray
    labels: np.ndarray
    objectives: tuple[float, float, float]
    rank: int = 0
    crowding: float = 0.0


def assign_by_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(squared_distances(x, centers), axis=1)


def make_individual(x: np.ndarray, k: int, centers: np.ndarray) -> Individual:
    labels = assign_by_centers(x, centers)
    for _ in range(5):
        new_centers = centers.copy()
        for j in range(k):
            pts = x[labels == j]
            if len(pts) > 0:
                new_centers[j] = pts.mean(axis=0)
        centers = new_centers
        labels = assign_by_centers(x, centers)
    labels = relabel_by_size(labels)
    comp = compactness(x, labels)
    sep = separation(x, labels)
    k_penalty = abs(k - 5) / 5.0
    return Individual(k=k, centers=centers, labels=labels, objectives=(comp, -sep, k_penalty))


def dominates(a: Individual, b: Individual) -> bool:
    return all(x <= y + 1e-12 for x, y in zip(a.objectives, b.objectives, strict=True)) and any(x < y - 1e-12 for x, y in zip(a.objectives, b.objectives, strict=True))


def non_dominated_sort(pop: list[Individual]) -> list[list[Individual]]:
    fronts: list[list[Individual]] = []
    dominated_by: dict[int, int] = {}
    dominates_list: dict[int, list[int]] = {}
    first = []
    for i, p in enumerate(pop):
        dominates_list[i] = []
        dominated_by[i] = 0
        for j, q in enumerate(pop):
            if i == j:
                continue
            if dominates(p, q):
                dominates_list[i].append(j)
            elif dominates(q, p):
                dominated_by[i] += 1
        if dominated_by[i] == 0:
            p.rank = 0
            first.append(i)
    current = first
    rank = 0
    while current:
        fronts.append([pop[i] for i in current])
        nxt = []
        for i in current:
            for j in dominates_list[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    pop[j].rank = rank + 1
                    nxt.append(j)
        rank += 1
        current = nxt
    return fronts


def crowding_distance(front: list[Individual]) -> None:
    if not front:
        return
    for ind in front:
        ind.crowding = 0.0
    m = len(front[0].objectives)
    for obj in range(m):
        front.sort(key=lambda ind: ind.objectives[obj])
        front[0].crowding = front[-1].crowding = math.inf
        lo, hi = front[0].objectives[obj], front[-1].objectives[obj]
        if abs(hi - lo) < 1e-12:
            continue
        for i in range(1, len(front) - 1):
            front[i].crowding += (front[i + 1].objectives[obj] - front[i - 1].objectives[obj]) / (hi - lo)


def nsga2_clustering(x: np.ndarray, *, seed: int = 42, pop_size: int = 40, generations: int = 35, k_min: int = 2, k_max: int = 6) -> Individual:
    rng = np.random.default_rng(seed)

    def random_individual() -> Individual:
        k = int(rng.integers(k_min, k_max + 1))
        idx = rng.choice(len(x), size=k, replace=False)
        centers = x[idx].astype(float).copy() + rng.normal(0, 0.05, size=(k, x.shape[1]))
        return make_individual(x, k, centers)

    def crossover(a: Individual, b: Individual) -> Individual:
        k = int(rng.choice([a.k, b.k, int(round((a.k + b.k) / 2))]))
        k = max(k_min, min(k_max, k))
        pool = np.vstack([a.centers, b.centers])
        idx = rng.choice(len(pool), size=k, replace=False if len(pool) >= k else True)
        centers = pool[idx].copy()
        centers += rng.normal(0, 0.08, size=centers.shape)
        if rng.random() < 0.25:
            centers[int(rng.integers(k))] = x[int(rng.integers(len(x)))]
        return make_individual(x, k, centers)

    pop = [random_individual() for _ in range(pop_size)]
    for _ in range(generations):
        fronts = non_dominated_sort(pop)
        for front in fronts:
            crowding_distance(front)
        mating = sorted(pop, key=lambda ind: (ind.rank, -ind.crowding))[: pop_size // 2]
        children = []
        while len(children) < pop_size:
            a, b = rng.choice(mating, size=2, replace=True)
            children.append(crossover(a, b))
        combined = pop + children
        fronts = non_dominated_sort(combined)
        new_pop = []
        for front in fronts:
            crowding_distance(front)
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                new_pop.extend(sorted(front, key=lambda ind: -ind.crowding)[: pop_size - len(new_pop)])
                break
        pop = new_pop
    fronts = non_dominated_sort(pop)
    first = fronts[0]
    # Select a balanced compromise: high silhouette, low compactness, close to known five drug categories.
    best = max(first, key=lambda ind: silhouette_score(x, ind.labels) - 0.08 * compactness(x, ind.labels) - 0.03 * abs(ind.k - 5))
    best.labels = relabel_by_size(best.labels)
    return best


def evaluate(x: np.ndarray, labels: np.ndarray, drug_labels: np.ndarray) -> dict[str, float | int]:
    return {
        "clusters": cluster_count(labels),
        "noise_ratio": float(np.mean(labels < 0)),
        "compactness": compactness(x, labels),
        "separation": separation(x, labels),
        "silhouette": silhouette_score(x, labels),
        "davies_bouldin": davies_bouldin_index(x, labels),
        "ari_vs_drug": adjusted_rand_index(drug_labels, labels),
        "nmi_vs_drug": normalized_mutual_info(drug_labels, labels),
        "purity_vs_drug": purity(drug_labels, labels),
    }


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout(pad=2)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_clusters(raw_x: np.ndarray, labels: np.ndarray, drugs: list[str], feature_pair: tuple[str, str], method: str, metrics_row: dict[str, float | int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    unique = sorted(set(labels.tolist()))
    markers = ["o", "s", "^", "D", "P"]
    drug_to_marker = {drug: markers[i % len(markers)] for i, drug in enumerate(DRUG_ORDER)}
    for lab in unique:
        mask_lab = labels == lab
        color = PALETTE["neutral"] if lab < 0 else CLUSTER_COLORS[lab % len(CLUSTER_COLORS)]
        name = "noise" if lab < 0 else f"cluster {lab}"
        for drug in DRUG_ORDER:
            mask = mask_lab & (np.array(drugs) == drug)
            if np.any(mask):
                ax.scatter(raw_x[mask, 0], raw_x[mask, 1], s=54, c=[color], marker=drug_to_marker[drug], edgecolors="black", linewidths=0.45, alpha=0.86, label=f"{name}, {drug}")
    handles, labels_text = ax.get_legend_handles_labels()
    compact = dict(zip(labels_text, handles, strict=False))
    ax.legend(compact.values(), compact.keys(), fontsize=6.5, ncol=2, loc="best", handletextpad=0.25, columnspacing=0.7)
    ax.set_xlabel(feature_pair[0])
    ax.set_ylabel(feature_pair[1])
    ax.set_title(f"{METHOD_NAMES[method]} on {feature_pair[0]}-{feature_pair[1]}")
    ax.grid(axis="both", linestyle="--", linewidth=0.7, color=PALETTE["neutral"], alpha=0.45)
    ax.tick_params(width=1.6, length=5)
    text = f"k={metrics_row['clusters']}  sil={metrics_row['silhouette']:.3f}  NMI={metrics_row['nmi_vs_drug']:.3f}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": PALETTE["neutral"], "linewidth": 0.8})
    save_figure(fig, out_path)
    plt.close(fig)


def run_pair(df: pd.DataFrame, feature_pair: tuple[str, str], out_fig_dir: Path, seed: int) -> list[dict[str, object]]:
    raw_x = np.column_stack([encode_feature(df[a], a) for a in feature_pair]).astype(float)
    x, _, _ = standardize(raw_x)
    drug_map = {drug: i for i, drug in enumerate(DRUG_ORDER)}
    drug_labels = df["Drug"].map(drug_map).astype(int).to_numpy()
    drugs = df["Drug"].astype(str).tolist()
    rows: list[dict[str, object]] = []

    runs: dict[str, np.ndarray] = {}
    runs["kmeans"] = kmeans(x, 5, seed=seed)[0]
    runs["fuzzy_cmeans"] = fuzzy_cmeans(x, 5, seed=seed)[0]
    eps, min_samples = choose_dbscan_params(x)
    runs["dbscan"] = dbscan(x, eps, min_samples)
    runs["agglomerative"] = agglomerative(x, 5)
    runs["nsga2"] = nsga2_clustering(x, seed=seed).labels

    for method, labels in runs.items():
        metrics_row = evaluate(x, labels, drug_labels)
        row = {
            "feature_x": feature_pair[0],
            "feature_y": feature_pair[1],
            "method": method,
            **metrics_row,
        }
        if method == "dbscan":
            row["eps"] = eps
            row["min_samples"] = min_samples
        rows.append(row)
        stem = f"{feature_pair[0]}__{feature_pair[1]}__{method}".replace("/", "_")
        plot_clusters(raw_x, labels, drugs, feature_pair, method, metrics_row, out_fig_dir / stem)
    return rows


def make_summary_plots(metrics: pd.DataFrame, out_fig_dir: Path) -> None:
    best = metrics.sort_values("nmi_vs_drug", ascending=False).groupby(["feature_x", "feature_y"], as_index=False).first()
    labels = [f"{r.feature_x}-{r.feature_y}" for r in best.itertuples()]
    colors = [METHOD_COLORS[str(m)] for m in best["method"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    bars = ax.bar(labels, best["nmi_vs_drug"], color=colors, edgecolor="black", linewidth=1.2)
    for bar, value in zip(bars, best["nmi_vs_drug"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Best NMI with Drug")
    ax.set_title("Best feature-pair correlation with drug labels")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(width=1.6, length=5)
    ax.set_ylim(0, max(0.72, float(best["nmi_vs_drug"].max()) + 0.08))
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=PALETTE["neutral"], alpha=0.45)
    method_handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m], ec="black", lw=1.0) for m in METHOD_ORDER]
    ax.legend(method_handles, [METHOD_NAMES[m] for m in METHOD_ORDER], fontsize=9, ncol=2, loc="upper right")
    save_figure(fig, out_fig_dir / "best_feature_pair_nmi")
    plt.close(fig)

    pivot = metrics.pivot_table(index="method", values=["silhouette", "nmi_vs_drug", "purity_vs_drug"], aggfunc="mean").loc[METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = np.arange(len(pivot.index))
    width = 0.25
    metric_colors = [PALETTE["blue_main"], PALETTE["green_3"], PALETTE["red_2"]]
    hatches = ["", "//", "\\\\"]
    for i, col in enumerate(["silhouette", "nmi_vs_drug", "purity_vs_drug"]):
        bars = ax.bar(x + (i - 1) * width, pivot[col], width=width, label=col, color=metric_colors[i], edgecolor="black", linewidth=1.2, hatch=hatches[i], alpha=0.92)
        for bar, value in zip(bars, pivot[col], strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in pivot.index], rotation=25, ha="right")
    ax.set_title("Average clustering quality across 10 feature pairs")
    ax.set_ylim(0, max(0.85, float(pivot.max().max()) + 0.1))
    ax.tick_params(width=1.6, length=5)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color=PALETTE["neutral"], alpha=0.45)
    ax.legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, out_fig_dir / "method_average_metrics")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drug200 clustering experiment 2 without package clustering calls.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "raw" / "drug200" / "drug200.csv"), help="Local CSV path or Hugging Face dataset repo.")
    parser.add_argument("--split", default=None, help="Optional Hugging Face split name.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "paper2-drug200-clustering")
    parser.add_argument("--fig-dir", type=Path, default=None, help="Defaults to <output-dir>/figures.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir
    fig_dir = args.fig_dir or output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_drug200_dataframe(args.dataset, split=args.split)
    rows: list[dict[str, object]] = []
    for pair in combinations(FEATURES, 2):
        rows.extend(run_pair(df, pair, fig_dir, args.seed))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "drug200_clustering_metrics.csv", index=False)
    best_by_pair = metrics.sort_values("nmi_vs_drug", ascending=False).groupby(["feature_x", "feature_y"], as_index=False).first()
    best_by_pair.to_csv(output_dir / "drug200_best_by_feature_pair.csv", index=False)
    method_summary = metrics.groupby("method")[["silhouette", "davies_bouldin", "ari_vs_drug", "nmi_vs_drug", "purity_vs_drug", "clusters"]].mean().reset_index()
    method_summary.to_csv(output_dir / "drug200_method_summary.csv", index=False)
    payload = {
        "dataset": str(args.dataset),
        "split": args.split,
        "feature_pairs": best_by_pair.to_dict(orient="records"),
        "method_summary": method_summary.to_dict(orient="records"),
    }
    (output_dir / "drug200_clustering_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_summary_plots(metrics, fig_dir)
    print("Saved metrics to", output_dir / "drug200_clustering_metrics.csv")
    print("Top feature pairs by best NMI:")
    print(best_by_pair[["feature_x", "feature_y", "method", "clusters", "silhouette", "nmi_vs_drug", "purity_vs_drug"]].sort_values("nmi_vs_drug", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nMethod summary:")
    print(method_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
