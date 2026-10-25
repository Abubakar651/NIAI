# Abubakar Iqbal
# NAVTTC Week 8 - Task 01: Simple RNN from Scratch

import numpy as np

class SimpleRNN:
    def __init__(self, in_dim, h_dim, out_dim, lr=0.01, seed=42):
        np.random.seed(seed)
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.lr = lr

        self.W_xh = np.random.randn(in_dim, h_dim) * 0.1
        self.W_hh = np.random.randn(h_dim, h_dim) * 0.1
        self.W_hy = np.random.randn(h_dim, out_dim) * 0.1
        self.b_h = np.zeros((1, h_dim))
        self.b_y = np.zeros((1, out_dim))

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, inputs):
        T = len(inputs)
        h_states = {-1: np.zeros((1, self.h_dim))}
        y_preds = {}

        for t in range(T):
            x_t = inputs[t]
            h_states[t] = np.tanh(np.dot(x_t, self.W_xh) + np.dot(h_states[t - 1], self.W_hh) + self.b_h)
            y_preds[t] = self.softmax(np.dot(h_states[t], self.W_hy) + self.b_y)
        return y_preds, h_states

    def train_step(self, inputs, targets):
        T = len(inputs)
        y_preds, h_states = self.forward(inputs)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        loss = 0.0
        for t in range(T):
            loss -= np.log(y_preds[t][0, targets[t]] + 1e-12)

        dh_next = np.zeros((1, self.h_dim))
        for t in reversed(range(T)):
            dy = np.copy(y_preds[t])
            dy[0, targets[t]] -= 1.0

            dW_hy += np.dot(h_states[t].T, dy)
            db_y += dy

            dh = np.dot(dy, self.W_hy.T) + dh_next
            dtanh = (1.0 - h_states[t] ** 2) * dh
            db_h += dtanh
            dW_xh += np.dot(inputs[t].T, dtanh)
            dW_hh += np.dot(h_states[t - 1].T, dtanh)
            dh_next = np.dot(dtanh, self.W_hh.T)

        for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(grad, -5.0, 5.0, out=grad)

        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.W_hy -= self.lr * dW_hy
        self.b_h -= self.lr * db_h
        self.b_y -= self.lr * db_y
        return loss / T

if __name__ == "__main__":
    print("NAVTTC Week 8 Task 01 - Abubakar Iqbal")

    abubakar_vocab = ["abubakar", "builds", "neural", "networks"]
    w2i = {w: i for i, w in enumerate(abubakar_vocab)}
    i2w = {i: w for i, w in enumerate(abubakar_vocab)}
    V = len(abubakar_vocab)

    X_train = [np.eye(V)[w2i[w]:w2i[w]+1] for w in ["abubakar", "builds", "neural"]]
    y_train = [w2i["builds"], w2i["neural"], w2i["networks"]]

    abubakar_rnn = SimpleRNN(in_dim=V, h_dim=8, out_dim=V, lr=0.1)

    for epoch in range(1, 151):
        loss = abubakar_rnn.train_step(X_train, y_train)
        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    preds, _ = abubakar_rnn.forward(X_train)
    print("\nNext Token Predictions:")
    for t in range(len(X_train)):
        pred_idx = np.argmax(preds[t])
        print(f"Input: '{abubakar_vocab[t]}' -> Predicted: '{i2w[pred_idx]}'")
