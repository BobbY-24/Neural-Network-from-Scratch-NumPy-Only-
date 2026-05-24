from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def relu(z: np.ndarray) -> np.ndarray:
    """Apply ReLU activation elementwise."""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Return the derivative mask for ReLU."""
    return z > 0


def softmax(z: np.ndarray) -> np.ndarray:
    """Compute column-wise softmax with numerical stabilization."""
    shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels into a one-hot matrix shaped `(classes, examples)`."""
    encoded = np.zeros((num_classes, labels.size))
    encoded[labels.astype(int), np.arange(labels.size)] = 1
    return encoded


def cross_entropy_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean cross-entropy loss for predicted probabilities."""
    m = labels.size
    clipped = np.clip(probs[labels.astype(int), np.arange(m)], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


class TwoLayerNeuralNetwork:
    """A two-layer fully connected neural network implemented with NumPy."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 10,
        output_dim: int = 10,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.params = Parameters(
            W1=rng.normal(0, 0.01, size=(hidden_dim, input_dim)),
            b1=np.zeros((hidden_dim, 1)),
            W2=rng.normal(0, 0.01, size=(output_dim, hidden_dim)),
            b2=np.zeros((output_dim, 1)),
        )

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run forward propagation.

        Args:
            x: Input matrix shaped `(features, examples)`.

        Returns:
            Hidden pre-activation, hidden activation, output pre-activation,
            and class probabilities.
        """
        z1 = self.params.W1 @ x + self.params.b1
        a1 = relu(z1)
        z2 = self.params.W2 @ a1 + self.params.b2
        a2 = softmax(z2)
        return z1, a1, z2, a2

    def backward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z1: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
    ) -> Parameters:
        """Run backpropagation and return parameter gradients."""
        m = x.shape[1]
        y_one_hot = one_hot(y, num_classes=self.params.W2.shape[0])
        dz2 = a2 - y_one_hot
        dW2 = (dz2 @ a1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = (self.params.W2.T @ dz2) * relu_derivative(z1)
        dW1 = (dz1 @ x.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        return Parameters(W1=dW1, b1=db1, W2=dW2, b2=db2)

    def update(self, gradients: Parameters, learning_rate: float) -> None:
        """Apply a gradient descent update."""
        self.params.W1 -= learning_rate * gradients.W1
        self.params.b1 -= learning_rate * gradients.b1
        self.params.W2 -= learning_rate * gradients.W2
        self.params.b2 -= learning_rate * gradients.b2

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for input examples."""
        *_, probs = self.forward(x)
        return np.argmax(probs, axis=0)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(x) == y))

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        iterations: int = 500,
        log_every: int = 50,
    ) -> list[dict[str, float]]:
        """Train the network with full-batch gradient descent."""
        history: list[dict[str, float]] = []
        for iteration in range(iterations):
            z1, a1, _, a2 = self.forward(x)
            gradients = self.backward(x, y, z1, a1, a2)
            self.update(gradients, learning_rate)
            if iteration % log_every == 0:
                history.append(
                    {
                        "iteration": float(iteration),
                        "loss": cross_entropy_loss(a2, y),
                        "accuracy": self.accuracy(x, y),
                    }
                )
        return history
