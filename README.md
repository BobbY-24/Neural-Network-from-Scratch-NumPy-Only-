# NumPy Neural Network For MNIST-Style Digit Classification

Two-layer neural network for MNIST-style digit classification implemented from scratch with NumPy.

This repository demonstrates the mechanics of neural networks below the framework level: forward propagation, backpropagation, ReLU, softmax, cross-entropy, gradient descent, data preprocessing, testing, and methodological documentation.

## Motivation

The goal is to show implementation-level understanding of basic neural network training without relying on PyTorch, TensorFlow, or scikit-learn model abstractions.

## What Is Included

- reusable NumPy model implementation
- command-line training entrypoint
- data loading and preprocessing helpers
- tests for core mathematical behavior
- methodology, limitations, and audit documentation
- preserved notebook from the original learning workflow

## Model

The network uses:

- 784 input features
- one hidden layer with ReLU activation
- 10-way softmax output
- cross-entropy loss
- full-batch gradient descent

## Preliminary Result

The original notebook reports preliminary development accuracy of `0.898` after 500 training iterations. This should be treated as a development result, not a benchmark claim, because the repository does not yet include repeated runs, confidence intervals, or baseline comparisons.

## Repository Structure

```text
.
├── README.md
├── src/
│   ├── neural_network.py
│   ├── train.py
│   └── utils.py
├── notebooks/
│   └── mnist_neural_network_from_scratch.ipynb
├── docs/
│   ├── audit.md
│   ├── methodology.md
│   └── limitations.md
├── data/
│   └── README.md
├── results/
│   └── README.md
├── tests/
│   └── test_neural_network.py
└── requirements.txt
```

## Reproducibility

```bash
git clone https://github.com/BobbY-24/Neural-Network-from-Scratch-NumPy-Only-.git
cd Neural-Network-from-Scratch-NumPy-Only-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Kaggle Digit Recognizer training CSV and place it at:

```text
data/train.csv
```

Train:

```bash
python -m src.train
```

Run tests:

```bash
pytest
```

## Limitations

- educational implementation rather than a production framework
- full-batch gradient descent is slow for larger datasets
- no repeated runs or confidence intervals yet
- no logistic regression, scikit-learn, or PyTorch baselines yet
- no robustness testing on noisy, rotated, shifted, or adversarial images yet

## Roadmap

- update notebook to import the reusable `src/` implementation
- add mini-batch gradient descent
- save training history to `results/`
- add baseline comparisons
- add robustness experiments
- add gradient-checking tests
