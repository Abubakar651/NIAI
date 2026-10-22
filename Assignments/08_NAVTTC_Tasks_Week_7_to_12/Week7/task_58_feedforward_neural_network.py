# Task 58 - Exercise on Feed Forward Neural Networks
# NAVTTC Outline Week 7 / Task 58
# Reference: https://builtin.com/data-science/feedforward-neural-network-intro

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class FeedForwardNeuralNetwork:
    """
    Modular Multi-Layer Feedforward Neural Network (FNN) from scratch.
    Features:
      - Custom Activation Functions: ReLU, Sigmoid, Softmax
      - Loss Function: Categorical Cross-Entropy
      - Backpropagation with Mini-Batch Gradient Descent
    """
    def __init__(self, layer_dims, learning_rate=0.05):
        self.layer_dims = layer_dims
        self.lr = learning_rate
        self.params = {}
        self.L = len(layer_dims) - 1

        # He-et-al initialization for ReLU hidden layers, Xavier for output layer
        np.random.seed(42)
        for l in range(1, self.L + 1):
            scale = np.sqrt(2.0 / layer_dims[l - 1]) if l < self.L else np.sqrt(1.0 / layer_dims[l - 1])
            self.params[f"W{l}"] = np.random.randn(layer_dims[l - 1], layer_dims[l]) * scale
            self.params[f"b{l}"] = np.zeros((1, layer_dims[l]))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        caches = {"A0": X}
        A = X
        # Hidden layers with ReLU
        for l in range(1, self.L):
            Z = np.dot(A, self.params[f"W{l}"]) + self.params[f"b{l}"]
            A = self.relu(Z)
            caches[f"Z{l}"] = Z
            caches[f"A{l}"] = A

        # Output layer with Softmax
        ZL = np.dot(A, self.params[f"W{self.L}"]) + self.params[f"b{self.L}"]
        AL = self.softmax(ZL)
        caches[f"Z{self.L}"] = ZL
        caches[f"A{self.L}"] = AL
        return AL, caches

    @staticmethod
    def compute_loss(Y_true_one_hot, Y_pred_prob):
        m = Y_true_one_hot.shape[0]
        # Avoid log(0)
        eps = 1e-12
        clipped_probs = np.clip(Y_pred_prob, eps, 1.0 - eps)
        loss = -np.sum(Y_true_one_hot * np.log(clipped_probs)) / m
        return loss

    def backward(self, Y_true_one_hot, caches):
        grads = {}
        m = Y_true_one_hot.shape[0]

        # Softmax with Categorical Cross-Entropy derivative: dZL = AL - Y
        dZL = caches[f"A{self.L}"] - Y_true_one_hot
        grads[f"dW{self.L}"] = np.dot(caches[f"A{self.L - 1}"].T, dZL) / m
        grads[f"db{self.L}"] = np.sum(dZL, axis=0, keepdims=True) / m

        dA_prev = np.dot(dZL, self.params[f"W{self.L}"].T)

        # Backpropagation through hidden layers
        for l in reversed(range(1, self.L)):
            dZ = dA_prev * self.relu_derivative(caches[f"Z{l}"])
            grads[f"dW{l}"] = np.dot(caches[f"A{l - 1}"].T, dZ) / m
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m
            if l > 1:
                dA_prev = np.dot(dZ, self.params[f"W{l}"].T)

        # Update weights and biases
        for l in range(1, self.L + 1):
            self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    def fit(self, X, y_one_hot, epochs=500, verbose=True):
        history = []
        for epoch in range(1, epochs + 1):
            AL, caches = self.forward(X)
            loss = self.compute_loss(y_one_hot, AL)
            history.append(loss)
            self.backward(y_one_hot, caches)

            if verbose and (epoch == 1 or epoch % 100 == 0 or epoch == epochs):
                preds = np.argmax(AL, axis=1)
                actuals = np.argmax(y_one_hot, axis=1)
                acc = np.mean(preds == actuals) * 100
                print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss:.5f} | Training Accuracy: {acc:.2f}%")
        return history

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


if __name__ == "__main__":
    print("=" * 65)
    print("Task 58: Feedforward Neural Network (FNN) Architecture")
    print("Student: Abubakar Iqbal | NAVTTC Week 7")
    print("=" * 65)

    # Multi-class spiral/circular dataset
    np.random.seed(42)
    N = 100  # Number of points per class
    D = 2    # Dimensionality
    K = 3    # Number of classes
    X = np.zeros((N * K, D))
    y = np.zeros(N * K, dtype="uint8")

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2 # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    # One-hot encoding of targets
    Y_one_hot = np.zeros((N * K, K))
    Y_one_hot[np.arange(N * K), y] = 1

    # Network topology: 2 inputs -> 16 (ReLU) -> 8 (ReLU) -> 3 (Softmax)
    topology = [2, 16, 8, 3]
    print(f"Constructing FNN with Topology: {topology}")
    fnn = FeedForwardNeuralNetwork(layer_dims=topology, learning_rate=0.2)
    
    loss_history = fnn.fit(X, Y_one_hot, epochs=600, verbose=True)

    test_preds = fnn.predict(X)
    final_acc = np.mean(test_preds == y) * 100
    print(f"\nFinal Classification Accuracy on Multi-class Spiral: {final_acc:.2f}%")
    print("Task 58 Feedforward Network Training Complete.")
