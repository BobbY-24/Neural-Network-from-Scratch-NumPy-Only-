# Methodology

## Data Source

I use the Kaggle Digit Recognizer / MNIST-style dataset. The model expects a CSV where the first column is the digit label and the remaining columns are 784 pixel values.

## Preprocessing

- Read the CSV into NumPy arrays.
- Split labels from pixel columns.
- Normalize pixel values by dividing by 255.
- Use the first 1,000 examples as a development set in the current workflow.

## Model

I implement a two-layer fully connected neural network:

- input layer: 784 pixel features,
- hidden layer: 10 units with ReLU activation,
- output layer: 10 units with softmax activation.

## Training

- Loss: cross-entropy.
- Optimizer: full-batch gradient descent.
- Default learning rate: `0.1`.
- Default iterations: `500`.

## Evaluation

I evaluate classification accuracy on a development split. The notebook currently reports preliminary development accuracy, but the repo does not yet include repeated runs or confidence intervals.

## Assumptions

- The input CSV follows the Kaggle Digit Recognizer format.
- Pixel values are in the range 0-255.
- A simple development split is acceptable for this educational implementation.
