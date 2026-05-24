# Neural Network from Scratch with NumPy

## Overview

I implemented a two-layer neural network for MNIST-style handwritten digit classification using NumPy only. The repository is now organized as a small ML foundations project with reusable source code, a training entrypoint, tests, documentation, and a notebook preserving the original learning workflow.

## Motivation

I built this project to understand neural networks below the framework level. Instead of relying on PyTorch or TensorFlow, I implemented forward propagation, backpropagation, gradient descent, ReLU, softmax, and cross-entropy directly with array operations.

## Objective

The goal of this project is to demonstrate a clear, reproducible NumPy-only implementation of a basic neural network for digit classification.

## Contributions

- Implemented a two-layer neural network with NumPy.
- Extracted reusable model, utility, and training code into `src/`.
- Added tests for softmax, forward-pass dimensions, finite loss, and dummy training.
- Documented data setup, methodology, limitations, and current results.
- Preserved the original notebook as part of the learning process.

## Repository Structure

```text
.
├── README.md
├── src/
│   ├── __init__.py
│   ├── neural_network.py
│   ├── train.py
│   └── utils.py
├── notebooks/
│   └── mnist_neural_network_from_scratch.ipynb
├── data/
│   └── README.md
├── results/
│   └── README.md
├── docs/
│   ├── audit.md
│   ├── methodology.md
│   └── limitations.md
├── tests/
│   └── test_neural_network.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Methodology

I use the Kaggle Digit Recognizer / MNIST-style CSV format. The preprocessing step separates labels from pixel columns and normalizes pixel values by dividing by 255.

The neural network uses:

- input layer with 784 pixel features,
- one hidden layer with ReLU activation,
- output layer with softmax activation,
- cross-entropy loss,
- full-batch gradient descent.

## Results

The original notebook reports a preliminary development accuracy of **0.898** after 500 training iterations. I label this as preliminary because I have not added repeated runs, confidence intervals, or baseline comparisons yet.

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

Train from the command line:

```bash
python -m src.train
```

Run tests:

```bash
pytest
```

Open the notebook:

```bash
jupyter notebook notebooks/mnist_neural_network_from_scratch.ipynb
```

## Limitations

- I use a simple architecture for educational clarity.
- The training loop is not optimized for speed.
- I have not added baseline comparisons against scikit-learn or PyTorch.
- The current evaluation uses a simple development split.
- The project does not test robustness to noisy, rotated, or shifted digit images.

## Future Work

- Update the notebook to import the reusable `src/` implementation.
- Add mini-batch gradient descent.
- Save training history to `results/`.
- Add baseline model comparisons.
- Add robustness experiments with perturbed digit images.
