# Neural Network from Scratch with NumPy

## Overview
I implemented a small neural network for handwritten digit classification using NumPy only. The notebook builds the core learning algorithm directly: forward propagation, activation functions, backpropagation, gradient descent, prediction, and evaluation.

## Motivation
I consider this one of my strongest portfolio projects because it shows implementation-level understanding instead of only library usage. Rebuilding a neural network from array operations helped me understand gradients, loss functions, parameter updates, and activation functions.

## Dataset
- **Source:** Kaggle Digit Recognizer / MNIST-style handwritten digit dataset.
- **File:** `data/mnist_train.zip`
- **Expected extracted file:** `data/train.csv`
- **Target variable:** digit label from `0` to `9`.
- **Important features:** flattened 28x28 grayscale pixel values.
- **Known limitations:** I use a clean benchmark dataset here and does not test harder vision settings such as noisy labels, rotations, or distribution shift.

## Methods
- I loaded handwritten digit data from CSV.
- I normalized pixel values for training.
- I implemented a two-layer neural network with NumPy.
- I implemented ReLU and softmax activations.
- I used one-hot encoding for labels.
- I implemented forward propagation, backpropagation, and gradient descent.
- I evaluated accuracy on a development set.

## Results
My notebook reports steady training improvement over 500 iterations and a final development set accuracy of **0.898**.

## Key Insights
- Implementing backpropagation directly made the learning process much more transparent.
- A small NumPy-only network can reach reasonable performance on a clean digit benchmark.
- Framework-free implementation is useful for learning, while larger experiments should use tested deep learning libraries.

## Limitations
- The implementation is educational and not optimized for speed or scale.
- I do not compare against scikit-learn or PyTorch baselines yet.
- The dataset must be extracted from `data/mnist_train.zip` before running.
- I do not test robustness to noise, rotations, or distribution shift.

## Future Improvements
- Extract reusable functions into `src/neural_network.py`.
- Add tests for activation functions, one-hot encoding, and gradient shapes.
- Compare performance against scikit-learn and PyTorch baselines.
- Add experiments on noisy or shifted digit images.

## How to Run
```bash
git clone https://github.com/BobbY-24/Neural-Network-from-Scratch-NumPy-Only-.git
cd Neural-Network-from-Scratch-NumPy-Only-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
unzip data/mnist_train.zip -d data/
jupyter notebook notebooks/neural_network_from_scratch.ipynb
```
