# Abubakar Iqbal
# NAVTTC Week 7 - Task 58: Feedforward Neural Network

import numpy as np

class FeedForwardNeuralNetwork:
    def __init__(self, layer_dims, lr=0.05, seed=42):
        self.layer_dims = layer_dims
        self.lr = lr
        self.params = {}
        self.L = len(layer_dims) - 1

        np.random.seed(seed)
        for l in range(1, self.L + 1):
            scale = np.sqrt(2.0 / layer_dims[l - 1]) if l < self.L else np.sqrt(1.0 / layer_dims[l - 1])
            self.params[f"W{l}"] = np.random.randn(layer_dims[l - 1], layer_dims[l]) * scale
            self.params[f"b{l}"] = np.zeros((1, layer_dims[l]))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        cache = {"A0": X}
        A = X
        for l in range(1, self.L):
            Z = np.dot(A, self.params[f"W{l}"]) + self.params[f"b{l}"]
            A = self.relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        ZL = np.dot(A, self.params[f"W{self.L}"]) + self.params[f"b{self.L}"]
        AL = self.softmax(ZL)
        cache[f"Z{self.L}"] = ZL
        cache[f"A{self.L}"] = AL
        return AL, cache

    def compute_loss(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        clipped = np.clip(Y_pred, 1e-12, 1.0 - 1e-12)
        return -np.sum(Y_true * np.log(clipped)) / m

    def backward(self, Y_true, cache):
        grads = {}
        m = Y_true.shape[0]
        AL = cache[f"A{self.L}"]
        dZL = AL - Y_true
        grads[f"dW{self.L}"] = np.dot(cache[f"A{self.L - 1}"].T, dZL) / m
        grads[f"db{self.L}"] = np.sum(dZL, axis=0, keepdims=True) / m

        dA = np.dot(dZL, self.params[f"W{self.L}"].T)
        for l in reversed(range(1, self.L)):
            dZ = dA * self.relu_deriv(cache[f"Z{l}"])
            grads[f"dW{l}"] = np.dot(cache[f"A{l - 1}"].T, dZ) / m
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m
            if l > 1:
                dA = np.dot(dZ, self.params[f"W{l}"].T)

        for l in range(1, self.L + 1):
            self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    def train(self, X, Y, epochs=300):
        for ep in range(1, epochs + 1):
            probs, cache = self.forward(X)
            loss = self.compute_loss(Y, probs)
            self.backward(Y, cache)
            if ep == 1 or ep % 100 == 0:
                print(f"Epoch {ep:3d} | Loss: {loss:.5f}")

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

if __name__ == "__main__":
    print("NAVTTC Week 7 Task 58 - Abubakar Iqbal")

    np.random.seed(42)
    N = 90
    abubakar_X0 = np.random.randn(N, 2) + np.array([-2, -2])
    abubakar_X1 = np.random.randn(N, 2) + np.array([2, 2])
    abubakar_X2 = np.random.randn(N, 2) + np.array([-2, 2])

    abubakar_X = np.vstack([abubakar_X0, abubakar_X1, abubakar_X2])
    abubakar_y = np.array([0] * N + [1] * N + [2] * N)

    Y_onehot = np.zeros((len(abubakar_y), 3))
    Y_onehot[np.arange(len(abubakar_y)), abubakar_y] = 1.0

    abubakar_fnn = FeedForwardNeuralNetwork([2, 16, 8, 3], lr=0.1)
    abubakar_fnn.train(abubakar_X, Y_onehot, epochs=300)

    abubakar_preds = abubakar_fnn.predict(abubakar_X)
    acc = np.mean(abubakar_preds == abubakar_y) * 100
    print(f"Abubakar's Multiclass Accuracy: {acc:.2f}%")
