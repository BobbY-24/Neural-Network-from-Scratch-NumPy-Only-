from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_mnist_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Kaggle Digit Recognizer CSV data.

    The expected format is one `label` column followed by 784 pixel columns.
    Returns `X` shaped `(784, examples)` and `y` shaped `(examples,)`.
    """
    data = pd.read_csv(path).to_numpy()
    y = data[:, 0].astype(int)
    x = data[:, 1:].T / 255.0
    return x, y


def train_dev_split(
    x: np.ndarray,
    y: np.ndarray,
    dev_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split normalized MNIST arrays into dev and train partitions."""
    if x.shape[1] <= dev_size:
        raise ValueError("dev_size must be smaller than the number of examples")
    x_dev, y_dev = x[:, :dev_size], y[:dev_size]
    x_train, y_train = x[:, dev_size:], y[dev_size:]
    return x_train, y_train, x_dev, y_dev
