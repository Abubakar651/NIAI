# Abubakar Iqbal
# NAVTTC Week 8 - Task 03: LSTM Sequence Prediction

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class LSTMSequencePredictor:
    def __init__(self, lookback=4, h_dim=8, lr=0.01, seed=42):
        self.lookback = lookback
        self.h_dim = h_dim
        self.lr = lr
        np.random.seed(seed)

        in_dim = 1 + h_dim
        self.W_f = np.random.randn(in_dim, h_dim) * 0.1
        self.b_f = np.ones((1, h_dim))
        self.W_i = np.random.randn(in_dim, h_dim) * 0.1
        self.b_i = np.zeros((1, h_dim))
        self.W_c = np.random.randn(in_dim, h_dim) * 0.1
        self.b_c = np.zeros((1, h_dim))
        self.W_o = np.random.randn(in_dim, h_dim) * 0.1
        self.b_o = np.zeros((1, h_dim))
        self.W_out = np.random.randn(h_dim, 1) * 0.1
        self.b_out = 0.0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

    def forward(self, seq):
        h = np.zeros((1, self.h_dim))
        c = np.zeros((1, self.h_dim))

        for val in seq:
            x_t = np.array([[val]])
            concat = np.hstack((x_t, h))
            f = self.sigmoid(np.dot(concat, self.W_f) + self.b_f)
            i = self.sigmoid(np.dot(concat, self.W_i) + self.b_i)
            c_bar = np.tanh(np.dot(concat, self.W_c) + self.b_c)
            c = f * c + i * c_bar
            o = self.sigmoid(np.dot(concat, self.W_o) + self.b_o)
            h = o * np.tanh(c)

        pred = np.dot(h, self.W_out) + self.b_out
        return float(pred[0, 0]), h

    def train_epoch(self, X_seqs, y_targets):
        losses = []
        for seq, target in zip(X_seqs, y_targets):
            pred, h = self.forward(seq)
            err = pred - target
            losses.append(err ** 2)

            grad_out = 2.0 * err
            self.W_out -= self.lr * grad_out * h.T
            self.b_out -= self.lr * grad_out
        return np.mean(losses)

if __name__ == "__main__":
    print("NAVTTC Week 8 Task 03 - Abubakar Iqbal")

    t = np.linspace(0, 8 * np.pi, 120)
    abubakar_series = np.sin(t) + 0.1 * np.random.RandomState(42).randn(120)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(abubakar_series.reshape(-1, 1)).flatten()

    lookback = 4
    X_data, y_data = [], []
    for i in range(len(scaled) - lookback):
        X_data.append(scaled[i : i + lookback])
        y_data.append(scaled[i + lookback])

    split = int(len(X_data) * 0.75)
    X_train, X_test = X_data[:split], X_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]

    abubakar_predictor = LSTMSequencePredictor(lookback=lookback, h_dim=8, lr=0.03)

    for ep in range(1, 41):
        loss = abubakar_predictor.train_epoch(X_train, y_train)
        if ep == 1 or ep % 10 == 0:
            print(f"Epoch {ep:2d} | Train MSE: {loss:.5f}")

    preds = [abubakar_predictor.forward(seq)[0] for seq in X_test]
    test_mse = mean_squared_error(y_test, preds)
    print(f"\nTest MSE: {test_mse:.5f}")

    print("\nNext 3 Forecasted Steps vs Actual:")
    for actual, pred in zip(y_test[:3], preds[:3]):
        print(f"Actual: {actual:.3f} | Predicted: {pred:.3f}")
