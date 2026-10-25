# Abubakar Iqbal
# NAVTTC Week 8 - Task 04: GRU Architecture from Scratch

import numpy as np

class GRUCell:
    def __init__(self, in_dim=4, h_dim=8, seed=42):
        np.random.seed(seed)
        self.in_dim = in_dim
        self.h_dim = h_dim

        self.W_z = np.random.randn(in_dim, h_dim) * 0.1
        self.U_z = np.random.randn(h_dim, h_dim) * 0.1
        self.b_z = np.zeros((1, h_dim))

        self.W_r = np.random.randn(in_dim, h_dim) * 0.1
        self.U_r = np.random.randn(h_dim, h_dim) * 0.1
        self.b_r = np.zeros((1, h_dim))

        self.W_h = np.random.randn(in_dim, h_dim) * 0.1
        self.U_h = np.random.randn(h_dim, h_dim) * 0.1
        self.b_h = np.zeros((1, h_dim))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def step(self, x_t, h_prev):
        z_t = self.sigmoid(np.dot(x_t, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z)
        r_t = self.sigmoid(np.dot(x_t, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r)
        h_cand = np.tanh(np.dot(x_t, self.W_h) + np.dot(r_t * h_prev, self.U_h) + self.b_h)
        h_t = (1.0 - z_t) * h_prev + z_t * h_cand
        return h_t, (z_t, r_t)

if __name__ == "__main__":
    print("NAVTTC Week 8 Task 04 - Abubakar Iqbal")

    abubakar_gru = GRUCell(in_dim=3, h_dim=4)
    abubakar_seq = np.array([
        [0.2, 0.5, -0.4],
        [0.8, -0.1, 0.3],
        [-0.5, 0.6, 0.1]
    ])

    h = np.zeros((1, 4))
    print("Sequential GRU Transitions:")
    for t, x in enumerate(abubakar_seq):
        x_t = x.reshape(1, -1)
        h, (z, r) = abubakar_gru.step(x_t, h)
        print(f"Step {t+1} | Mean Update Gate: {np.mean(z):.3f} | Mean Reset Gate: {np.mean(r):.3f}")
        print(f"       Hidden State: {h[0].round(3)}")
