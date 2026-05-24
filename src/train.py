from __future__ import annotations

import argparse
from pathlib import Path

from .neural_network import TwoLayerNeuralNetwork
from .utils import load_mnist_csv, train_dev_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy-only MNIST neural network.")
    parser.add_argument("--data", default="data/train.csv", help="Path to MNIST-style CSV file.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--dev-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. See data/README.md for download instructions."
        )

    x, y = load_mnist_csv(data_path)
    x_train, y_train, x_dev, y_dev = train_dev_split(x, y, dev_size=args.dev_size)

    model = TwoLayerNeuralNetwork(seed=42)
    history = model.train(
        x_train,
        y_train,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
    )

    for row in history:
        print(
            f"iteration={int(row['iteration'])} "
            f"loss={row['loss']:.4f} "
            f"train_accuracy={row['accuracy']:.4f}"
        )
    print(f"dev_accuracy={model.accuracy(x_dev, y_dev):.4f}")


if __name__ == "__main__":
    main()
