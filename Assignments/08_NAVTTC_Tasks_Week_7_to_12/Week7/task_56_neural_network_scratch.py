# Abubakar Iqbal
# NAVTTC Week 7 - Task 56: Neural Network from Scratch

import numpy as np

class NeuralNetworkFromScratch:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1, lr=0.1, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.losses = []

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    def sigmoid_deriv(self, a):
        return a * (1.0 - a)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, out):
        m = X.shape[0]
        dz2 = (out - y) * self.sigmoid_deriv(out)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_deriv(self.a1)
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        for epoch in range(1, epochs + 1):
            out = self.forward(X)
            loss = np.mean((y - out) ** 2)
            self.losses.append(loss)
            self.backward(X, y, out)
            if epoch == 1 or epoch % 250 == 0:
                print(f"Epoch {epoch:4d} | MSE: {loss:.5f}")

    def predict(self, X):
        probs = self.forward(X)
        return (probs >= 0.5).astype(int), probs

if __name__ == "__main__":
    print("NAVTTC Week 7 Task 56 - Abubakar Iqbal")

    abubakar_X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)
    abubakar_y = np.array([[0], [1], [1], [0]], dtype=float)

    abubakar_nn = NeuralNetworkFromScratch(input_dim=2, hidden_dim=4, output_dim=1, lr=0.5)
    abubakar_nn.train(abubakar_X, abubakar_y, epochs=1000)

    abubakar_preds, abubakar_probs = abubakar_nn.predict(abubakar_X)
    print("\nAbubakar's XOR Predictions:")
    for i in range(len(abubakar_X)):
        print(f"Input: {abubakar_X[i]} -> Target: {int(abubakar_y[i][0])} | Pred: {abubakar_preds[i][0]} (Prob: {abubakar_probs[i][0]:.4f})")

    acc = np.mean(abubakar_preds == abubakar_y) * 100
    print(f"Accuracy: {acc:.1f}%")
