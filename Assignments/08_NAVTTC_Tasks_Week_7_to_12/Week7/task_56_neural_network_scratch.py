# Task 56 - Demonstration of Neural Networks from Scratch in Python
# NAVTTC Outline Week 7 / Task 56
# Reference: https://www.analyticsvidhya.com/blog/2020/07/neural-networks-from-scratch-in-pythonand-r

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class NeuralNetworkFromScratch:
    """
    A 2-Layer Feedforward Neural Network built from scratch with NumPy.
    Architecture: Input Layer -> Hidden Layer (Sigmoid) -> Output Layer (Sigmoid)
    Implements: Forward propagation, Mean Squared Error Loss, and Backpropagation with Gradient Descent.
    """
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.1, random_state=42):
        np.random.seed(random_state)
        self.lr = learning_rate
        # Xavier/He normal weight initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1.0 - a)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, output):
        m = X.shape[0]
        # Output layer gradients (MSE Loss)
        dZ2 = (output - y) * self.sigmoid_derivative(output)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Parameter updates via Gradient Descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000, verbose=True):
        for epoch in range(1, epochs + 1):
            out = self.forward(X)
            loss = np.mean((y - out) ** 2)
            self.loss_history.append(loss)
            self.backward(X, y, out)
            if verbose and (epoch == 1 or epoch % 200 == 0 or epoch == epochs):
                print(f"Epoch {epoch:4d}/{epochs} | MSE Loss: {loss:.6f}")

    def predict(self, X):
        probs = self.forward(X)
        return (probs >= 0.5).astype(int), probs


if __name__ == "__main__":
    print("=" * 65)
    print("Task 56: 2-Layer Neural Network from Scratch (NumPy Implementation)")
    print("Student: Abubakar Iqbal | NAVTTC Week 7")
    print("=" * 65)

    # Classic non-linear XOR Problem (Cannot be solved by a single perceptron)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    print("\nTraining on XOR Truth Table:")
    print("Input Features (X):\n", X)
    print("Target Labels (y):\n", y)

    nn = NeuralNetworkFromScratch(input_dim=2, hidden_dim=4, output_dim=1, learning_rate=0.5, random_state=42)
    nn.train(X, y, epochs=1000, verbose=True)

    preds, raw_probs = nn.predict(X)
    print("\n--- Model Evaluation & Final Predictions ---")
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Target: {int(y[i][0])} | Predicted Prob: {raw_probs[i][0]:.4f} | Predicted Class: {preds[i][0]}")

    accuracy = np.mean(preds == y) * 100
    print(f"\nFinal XOR Classification Accuracy: {accuracy:.1f}%")
    print("Task 56 Execution Completed Successfully.")
