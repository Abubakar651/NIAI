# Abubakar Iqbal
# NAVTTC Week 10 - Task 01: Bi-Directional LSTM from Scratch

import numpy as np

class BiDirectionalLSTM:
    def __init__(self, in_dim=4, h_dim=8, seed=42):
        np.random.seed(seed)
        self.in_dim = in_dim
        self.h_dim = h_dim
        concat_dim = in_dim + h_dim

        self.W_fwd = np.random.randn(concat_dim, 4 * h_dim) * 0.1
        self.b_fwd = np.zeros((1, 4 * h_dim))
        self.b_fwd[0, :h_dim] = 1.0

        self.W_bwd = np.random.randn(concat_dim, 4 * h_dim) * 0.1
        self.b_bwd = np.zeros((1, 4 * h_dim))
        self.b_bwd[0, :h_dim] = 1.0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def _step(self, x_t, h_prev, c_prev, W, b):
        concat = np.hstack((x_t, h_prev))
        gates = np.dot(concat, W) + b
        D = self.h_dim

        f = self.sigmoid(gates[:, :D])
        i = self.sigmoid(gates[:, D:2*D])
        c_bar = np.tanh(gates[:, 2*D:3*D])
        o = self.sigmoid(gates[:, 3*D:])

        c_next = f * c_prev + i * c_bar
        h_next = o * np.tanh(c_next)
        return h_next, c_next

    def forward(self, seq):
        T = len(seq)
        fwd_states = []
        bwd_states = [None] * T

        h_f = np.zeros((1, self.h_dim))
        c_f = np.zeros((1, self.h_dim))
        for t in range(T):
            h_f, c_f = self._step(seq[t:t+1], h_f, c_f, self.W_fwd, self.b_fwd)
            fwd_states.append(h_f)

        h_b = np.zeros((1, self.h_dim))
        c_b = np.zeros((1, self.h_dim))
        for t in reversed(range(T)):
            h_b, c_b = self._step(seq[t:t+1], h_b, c_b, self.W_bwd, self.b_bwd)
            bwd_states[t] = h_b

        fused = [np.hstack((fwd_states[t], bwd_states[t])) for t in range(T)]
        return np.vstack(fused)

if __name__ == "__main__":
    print("NAVTTC Week 10 Task 01 - Abubakar Iqbal")

    abubakar_bilstm = BiDirectionalLSTM(in_dim=3, h_dim=4)
    abubakar_seq = np.array([
        [0.8, -0.2, 0.4],
        [0.1,  0.9, 0.3],
        [-0.4, 0.5, 0.7],
        [0.6, -0.1, 0.2]
    ])

    fused_repr = abubakar_bilstm.forward(abubakar_seq)
    print(f"Input Sequence Shape:  {abubakar_seq.shape}")
    print(f"BiLSTM Output Shape:   {fused_repr.shape} (Fused Forward + Backward)")
    print("\nRepresentation at each step:")
    for t in range(len(abubakar_seq)):
        print(f"Step {t+1}: {fused_repr[t].round(3)}")
