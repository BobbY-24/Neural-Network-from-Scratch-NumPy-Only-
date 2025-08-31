🧠 Neural Network from Scratch (NumPy Only)
This project is a hand-built neural network for handwritten digit classification on the MNIST dataset, inspired by the YouTube video “Building a Neural Network FROM SCRATCH (no Tensorflow/PyTorch, just NumPy & math)” by Samson Zhang.
The goal is to fully understand the mechanics of forward propagation, backward propagation, gradient descent, and prediction without relying on deep learning frameworks like TensorFlow or PyTorch.

🚀 Features
Implements a 2-layer neural network using only NumPy.


Covers:


Forward propagation


Backward propagation


Weight updates via gradient descent


Activation functions: ReLU, Softmax, Sigmoid, Tanh


Loss calculation (Cross-Entropy Loss)


Prediction and evaluation functions


Visualizes predictions with matplotlib.


Supports showing multiple random test samples with predicted labels.



📊 Training Results
Example training output:
Iteration 0   → Training Accuracy: 0.0924
Iteration 50  → Training Accuracy: 0.2452
Iteration 100 → Training Accuracy: 0.4828
Iteration 150 → Training Accuracy: 0.7061
Iteration 200 → Training Accuracy: 0.7871
Iteration 250 → Training Accuracy: 0.8231
Iteration 300 → Training Accuracy: 0.8472
Iteration 350 → Training Accuracy: 0.8653
Iteration 400 → Training Accuracy: 0.8788
Iteration 450 → Training Accuracy: 0.8868

✅ Final Dev Set Accuracy: ~89.8%

📂 Project Structure
├── neural_network.py   # main implementation
├── utils.py            # helper functions (prediction, visualization, etc.)
├── README.md           # project description
└── data/               # MNIST dataset (train & dev set)


⚙️ How It Works
Input Layer: Flattens 28x28 MNIST images → 784 features.


Hidden Layer: Fully connected, ReLU activation.


Output Layer: 10 neurons, Softmax activation (digits 0–9).


Training: Uses cross-entropy loss and gradient descent.


Evaluation: Accuracy on both training and dev set.


Visualization: Display random predictions with matplotlib.



🔧 Requirements
Python 3.x


NumPy


Matplotlib


Install dependencies:
pip install numpy matplotlib


▶️ Run the Project
python neural_network.py

To visualize predictions:
test_multiple_predictions(W1, b1, W2, b2, num_images=5)


🧩 Key Concepts Learned
Activation functions (ReLU, Sigmoid, Tanh, Softmax)


One-hot encoding for labels


Forward and backward propagation


Gradient descent parameter updates


Loss functions and optimization


Cross-validation for model evaluation



🙌 Acknowledgments
Tutorial by Samson Zhang on YouTube.


MNIST dataset.




