from __future__ import annotations

import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset


def _flatten_mnist_split(split) -> TensorDataset:
    images = torch.stack(
        [torch.from_numpy(np.asarray(image, dtype=np.float32)).view(28 * 28) / 255.0 for image in split["image"]]
    )
    labels = torch.tensor(split["label"], dtype=torch.long)
    return TensorDataset(images, labels)


def mnist_loaders(
    batch_size: int,
    eval_batch_size: int,
    limit_train: int | None = None,
    limit_test: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    dataset = load_dataset("ylecun/mnist")
    train_split = dataset["train"]
    test_split = dataset["test"]
    if limit_train is not None:
        train_split = train_split.select(range(min(limit_train, len(train_split))))
    if limit_test is not None:
        test_split = test_split.select(range(min(limit_test, len(test_split))))

    train_data = _flatten_mnist_split(train_split)
    test_data = _flatten_mnist_split(test_split)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
