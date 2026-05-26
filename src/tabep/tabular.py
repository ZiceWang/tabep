from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TabularDatasetBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    input_size: int
    output_size: int


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _dataset_to_dataframe(dataset: Dataset | DatasetDict) -> pd.DataFrame:
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            dataset = dataset[next(iter(dataset.keys()))]
    return dataset.to_pandas()


def load_drug200_dataframe(source: str | Path, *, split: str | None = None) -> pd.DataFrame:
    """Load Drug200 from a local CSV path or a Hugging Face dataset repo."""
    source_str = str(source)
    source_path = Path(source_str)
    if source_path.exists() or source_path.suffix.lower() == ".csv":
        return pd.read_csv(source_path)

    load_kwargs: dict[str, Any] = {}
    if split is not None:
        load_kwargs["split"] = split
    dataset = load_dataset(source_str, **load_kwargs)
    return _dataset_to_dataframe(dataset)


def load_ucirepo_dataframe(dataset_id: int) -> tuple[pd.DataFrame, str, str]:
    """Load a UCI ML Repository dataset through ``ucimlrepo``.

    Returns the joined dataframe, target-column name, and official dataset name.
    """
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=dataset_id)
    features = dataset.data.features
    targets = dataset.data.targets
    if features is None or targets is None:
        raise ValueError(f"UCI dataset id={dataset_id} does not expose both features and targets through ucimlrepo.")
    if targets.shape[1] != 1:
        raise ValueError(f"UCI dataset id={dataset_id} has {targets.shape[1]} target columns; expected exactly one.")
    target_column = str(targets.columns[0])
    df = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    return df, target_column, str(dataset.metadata.name)


def _build_tabular_bundle(
    df: pd.DataFrame,
    *,
    target_column: str,
    test_size: float,
    seed: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    max_samples: int | None = None,
) -> TabularDatasetBundle:
    if target_column not in df.columns:
        raise ValueError(f"target column {target_column!r} not found")

    df = df.copy()
    df = df.dropna(subset=[target_column])
    x_df = df.drop(columns=[target_column])
    y_raw = df[target_column].astype(str).str.strip().str.removesuffix(".").to_numpy()
    values, counts = np.unique(y_raw, return_counts=True)
    if np.any(counts < 2):
        keep_values = set(values[counts >= 2])
        keep_mask = np.asarray([value in keep_values for value in y_raw])
        x_df = x_df.loc[keep_mask].reset_index(drop=True)
        y_raw = y_raw[keep_mask]

    numeric_features = x_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in x_df.columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_one_hot_encoder())]), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    if max_samples is not None and max_samples > 0 and y.size > max_samples:
        sample_idx, _ = train_test_split(
            np.arange(y.size),
            train_size=max_samples,
            random_state=seed,
            stratify=y,
        )
        sample_idx = np.sort(sample_idx)
        x_df = x_df.iloc[sample_idx].reset_index(drop=True)
        y = y[sample_idx]

    x_train_df, x_test_df, y_train, y_test = train_test_split(
        x_df,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    x_train = preprocessor.fit_transform(x_train_df).astype(np.float32)
    x_test = preprocessor.transform(x_test_df).astype(np.float32)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:  # pragma: no cover
        feature_names = [f"feature_{idx}" for idx in range(x_train.shape[1])]

    pin_memory = torch.cuda.is_available()
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64)))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test.astype(np.int64)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return TabularDatasetBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=feature_names,
        class_names=label_encoder.classes_.tolist(),
        input_size=int(x_train.shape[1]),
        output_size=int(len(label_encoder.classes_)),
    )


def load_drug200_bundle(
    source: str | Path,
    *,
    target_column: str = "Drug",
    split: str | None = None,
    test_size: float = 0.8,
    seed: int = 42,
    batch_size: int = 32,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    max_samples: int | None = None,
) -> TabularDatasetBundle:
    """Load Drug200 as a small tabular classification benchmark.

    Numerical columns are standardized and categorical columns are one-hot encoded,
    which keeps the feature interface simple for both TabEP and sklearn baselines.
    """
    return _build_tabular_bundle(
        load_drug200_dataframe(source, split=split),
        target_column=target_column,
        test_size=test_size,
        seed=seed,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )


def load_ucirepo_bundle(
    dataset_id: int,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    batch_size: int = 32,
    eval_batch_size: int = 256,
    num_workers: int = 0,
    max_samples: int | None = None,
) -> tuple[TabularDatasetBundle, str]:
    """Load a classification dataset from UCI through ``ucimlrepo``."""
    df, target_column, dataset_name = load_ucirepo_dataframe(dataset_id)
    bundle = _build_tabular_bundle(
        df,
        target_column=target_column,
        test_size=test_size,
        seed=seed,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )
    return bundle, dataset_name


def make_sklearn_preprocessed_drug200(
    source: str | Path,
    *,
    target_column: str = "Drug",
    split: str | None = None,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return preprocessed arrays for non-neural baselines."""
    bundle = load_drug200_bundle(
        source,
        target_column=target_column,
        split=split,
        test_size=test_size,
        seed=seed,
        batch_size=32,
        eval_batch_size=256,
    )
    return bundle.x_train, bundle.x_test, bundle.y_train, bundle.y_test, bundle.class_names
