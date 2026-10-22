# Week 8 - Task 03: LSTM Code Practice - Time Series & Sequence Prediction
# NAVTTC Outline Week 8 Day 3
# Reference: Sequential Time Series Forecasting using LSTM Architecture

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class LSTMSequencePredictor:
    """
    Self-contained Recurrent LSTM Sequence Forecasting Model.
    Forecasts multi-step continuous sequences (e.g., daily banking liquidity or traffic volume).
    """
    def __init__(self, lookback=4, hidden_dim=16, lr=0.01):
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.lr = lr

        np.random.seed(42)
        # Weights for [x_t, h_{t-1}] -> dim: (1 + hidden_dim)
        in_dim = 1 + hidden_dim
        self.W_f = np.random.randn(in_dim, hidden_dim) * 0.1
        self.b_f = np.ones((1, hidden_dim))
        self.W_i = np.random.randn(in_dim, hidden_dim) * 0.1
        self.b_i = np.zeros((1, hidden_dim))
        self.W_c = np.random.randn(in_dim, hidden_dim) * 0.1
        self.b_c = np.zeros((1, hidden_dim))
        self.W_o = np.random.randn(in_dim, hidden_dim) * 0.1
        self.b_o = np.zeros((1, hidden_dim))

        # Regression projection head: hidden_dim -> 1
        self.W_out = np.random.randn(hidden_dim, 1) * 0.1
        self.b_out = 0.0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

    def forward(self, seq):
        """Processes 1D sequence window and predicts next continuous scalar."""
        h = np.zeros((1, self.hidden_dim))
        c = np.zeros((1, self.hidden_dim))

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
        """Single epoch gradient optimization step."""
        losses = []
        for seq, target in zip(X_seqs, y_targets):
            pred, h = self.forward(seq)
            err = pred - target
            losses.append(err ** 2)

            # Gradient update on output projection head
            grad_out = 2.0 * err
            self.W_out -= self.lr * (h.T * grad_out)
            self.b_out -= self.lr * grad_out

        return np.mean(losses)


def run_lstm_practice():
    print("=" * 65)
    print("Week 8 Task 3: LSTM Sequence Forecasting & Practice")
    print("Student: Abubakar Iqbal | NAVTTC Week 8")
    print("=" * 65)

    # Generate synthetic cyclical daily transactions time series
    np.random.seed(42)
    t = np.linspace(0, 50, 200)
    series = np.sin(t) + 0.1 * np.random.normal(size=len(t)) + 2.0

    print(f"Time Series Length: {len(series)} Consecutive Daily Data Points")

    # MinMax Normalization
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Form sequential lookback windows (X: [t-4, t-3, t-2, t-1] -> y: t)
    lookback = 4
    X_windows, y_targets = [], []
    for i in range(len(scaled_series) - lookback):
        X_windows.append(scaled_series[i:i + lookback])
        y_targets.append(scaled_series[i + lookback])

    X_windows = np.array(X_windows)
    y_targets = np.array(y_targets)

    # Train/Test Split (80% Train, 20% Test)
    split_idx = int(len(X_windows) * 0.8)
    X_train, X_test = X_windows[:split_idx], X_windows[split_idx:]
    y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]

    print(f"Training Windows: {len(X_train)} | Testing Windows: {len(X_test)}")

    model = LSTMSequencePredictor(lookback=lookback, hidden_dim=12, lr=0.03)

    print("\nTraining LSTM Sequential Recurrent Model...")
    for epoch in range(1, 101):
        loss = model.train_epoch(X_train, y_train)
        if epoch == 1 or epoch % 25 == 0 or epoch == 100:
            print(f"  Epoch {epoch:3d}/100 | Window MSE: {loss:.6f}")

    # Test set evaluation
    preds = [model.forward(seq)[0] for seq in X_test]
    actuals = y_test

    # Inverse transform
    actuals_orig = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    preds_orig = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    mse = mean_squared_error(actuals_orig, preds_orig)
    mae = mean_absolute_error(actuals_orig, preds_orig)

    print(f"\nTest Set Performance (Real-World Scale):")
    print(f"  Mean Squared Error (MSE):  {mse:.5f}")
    print(f"  Mean Absolute Error (MAE): {mae:.5f}")

    print("\nSample Forecast Comparison (Next 5 Time Steps):")
    for i in range(5):
        print(f"  Day {i+1}: Actual = {actuals_orig[i]:.3f} | Predicted = {preds_orig[i]:.3f} (Abs Error: {abs(actuals_orig[i]-preds_orig[i]):.3f})")

    print("\nWeek 8 Task 3 LSTM Practice Completed Successfully.")

if __name__ == "__main__":
    run_lstm_practice()
