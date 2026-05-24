import numpy as np

from src.neural_network import (
    TwoLayerNeuralNetwork,
    cross_entropy_loss,
    softmax,
)


def test_softmax_columns_sum_to_one():
    z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    probs = softmax(z)
    assert np.allclose(probs.sum(axis=0), np.ones(2))


def test_forward_pass_dimensions():
    model = TwoLayerNeuralNetwork(input_dim=4, hidden_dim=3, output_dim=2)
    x = np.ones((4, 5))
    z1, a1, z2, a2 = model.forward(x)
    assert z1.shape == (3, 5)
    assert a1.shape == (3, 5)
    assert z2.shape == (2, 5)
    assert a2.shape == (2, 5)


def test_loss_is_finite():
    probs = np.array([[0.8, 0.2], [0.2, 0.8]])
    labels = np.array([0, 1])
    assert np.isfinite(cross_entropy_loss(probs, labels))


def test_training_step_runs_on_dummy_data():
    model = TwoLayerNeuralNetwork(input_dim=4, hidden_dim=3, output_dim=2)
    x = np.random.default_rng(0).normal(size=(4, 6))
    y = np.array([0, 1, 0, 1, 0, 1])
    history = model.train(x, y, iterations=2, log_every=1)
    assert len(history) == 2
