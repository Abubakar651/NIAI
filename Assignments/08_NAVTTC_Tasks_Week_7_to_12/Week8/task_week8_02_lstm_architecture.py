# Abubakar Iqbal
# NAVTTC Week 8 - Task 02: LSTM Architecture from Scratch

import numpy as np

class LSTMCell:
    def __init__(self, in_dim=4, h_dim=8, seed=42):
        np.random.seed(seed)
        self.in_dim = in_dim
        self.h_dim = h_dim
        concat_dim = in_dim + h_dim

        self.W_f = np.random.randn(concat_dim, h_dim) * 0.1
        self.b_f = np.ones((1, h_dim))

        self.W_i = np.random.randn(concat_dim, h_dim) * 0.1
        self.b_i = np.zeros((1, h_dim))

        self.W_c = np.random.randn(concat_dim, h_dim) * 0.1
        self.b_c = np.zeros((1, h_dim))

        self.W_o = np.random.randn(concat_dim, h_dim) * 0.1
        self.b_o = np.zeros((1, h_dim))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def step(self, x_t, h_prev, c_prev):
        concat = np.hstack((x_t, h_prev))

        f_t = self.sigmoid(np.dot(concat, self.W_f) + self.b_f)
        i_t = self.sigmoid(np.dot(concat, self.W_i) + self.b_i)
        c_bar = np.tanh(np.dot(concat, self.W_c) + self.b_c)
        c_t = f_t * c_prev + i_t * c_bar
        o_t = self.sigmoid(np.dot(concat, self.W_o) + self.b_o)
        h_t = o_t * np.tanh(c_t)

        return h_t, c_t, (f_t, i_t, o_t)

if __name__ == "__main__":
    print("NAVTTC Week 8 Task 02 - Abubakar Iqbal")

    abubakar_lstm = LSTMCell(in_dim=3, h_dim=4)

    abubakar_seq = np.array([
        [0.5, 0.2, -0.1],
        [0.1, 0.9,  0.4],
        [-0.3, 0.4, 0.8],
        [0.7, -0.2, 0.3]
    ])

    h = np.zeros((1, 4))
    c = np.zeros((1, 4))

    print("Sequential LSTM Gate Activity:")
    for t, x in enumerate(abubakar_seq):
        x_t = x.reshape(1, -1)
        h, c, (f, i, o) = abubakar_lstm.step(x_t, h, c)
        print(f"Step {t+1} | Mean Forget Gate: {np.mean(f):.3f} | Input Gate: {np.mean(i):.3f} | Output Gate: {np.mean(o):.3f}")
        print(f"       Hidden State: {h[0].round(3)}")
