# Neural Network from Scratch with NumPy

## Overview
This project implements a small neural network for handwritten digit classification using NumPy only. The notebook builds the core learning algorithm directly: forward propagation, activation functions, backpropagation, gradient descent, prediction, and evaluation. The goal is to understand the mechanics behind neural networks without relying on TensorFlow or PyTorch.

## Motivation
For an AI research-oriented portfolio, this project is valuable because it demonstrates implementation-level understanding rather than only library usage. Rebuilding a neural network from basic array operations helps clarify gradients, loss functions, parameter updates, and the role of activation functions. This is a useful foundation for later work in deep learning, robustness, and model evaluation.

## Dataset
- **Source:** Kaggle Digit Recognizer / MNIST-style handwritten digit dataset.
- **File:** `data/mnist_train.zip`
- **Expected extracted file:** `data/train.csv`
- **Target variable:** digit label from `0` to `9`.
- **Important features:** flattened 28x28 grayscale pixel values.
- **Dataset size:** TODO: add dataset size after rerunning notebook.
- **Known limitations:** This is a clean benchmark dataset and does not reflect harder real-world vision settings such as distribution shift, noisy labels, rotated digits, or adversarial examples.

## Methods
- Loaded handwritten digit data from CSV.
- Normalized pixel values for training.
- Implemented a two-layer neural network with NumPy.
- Implemented ReLU and softmax activations.
- Used one-hot encoding for labels.
- Implemented forward propagation, backpropagation, and gradient descent.
- Evaluated model accuracy on a development set.
- Visualized predictions with matplotlib.

## Results
The notebook reports the following training progression:

| Iteration | Training Accuracy |
| ---: | ---: |
| 0 | 0.0924 |
| 50 | 0.2452 |
| 100 | 0.4828 |
| 150 | 0.7061 |
| 200 | 0.7871 |
| 250 | 0.8231 |
| 300 | 0.8472 |
| 350 | 0.8653 |
| 400 | 0.8788 |
| 450 | 0.8868 |

Final reported development set accuracy: **0.898**.

## Key Insights
- Implementing backpropagation directly makes the learning process more transparent.
- A small NumPy-only network can reach reasonable performance on a clean digit benchmark.
- Accuracy improves steadily over training iterations, showing that the gradient updates are working.
- Framework-free implementation is useful for learning, but modern deep learning work should use tested libraries for larger experiments.

## Limitations
- The implementation is educational and not optimized for speed or scale.
- The project does not yet compare against logistic regression, scikit-learn baselines, or PyTorch implementations.
- The dataset must be manually extracted from `data/mnist_train.zip` before running.
- The notebook uses a single split and does not report confidence intervals.
- The project does not test robustness to noise, rotations, or distribution shift.

## Future Improvements
- Extract reusable functions into `src/neural_network.py`.
- Add tests for activation functions, one-hot encoding, and gradient shapes.
- Compare performance against scikit-learn and PyTorch baselines.
- Add experiments on noisy or shifted digit images.
- Add a concise technical writeup explaining the math behind backpropagation.

## How to Run
```bash
git clone https://github.com/BobbY-24/Neural-Network-from-Scratch-NumPy-Only-.git
cd Neural-Network-from-Scratch-NumPy-Only-
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
unzip data/mnist_train.zip -d data/
jupyter notebook notebooks/neural_network_from_scratch.ipynb
```

Run the notebook cells from top to bottom. The notebook expects the extracted dataset at `data/train.csv`.
