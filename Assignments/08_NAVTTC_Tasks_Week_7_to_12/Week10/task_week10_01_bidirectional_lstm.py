# Week 10 - Task 01: Bi-Directional LSTM / RNN in Sequence Models
# NAVTTC Outline Week 10 Day 1
# Reference: Schuster & Paliwal (1997) / Bidirectional Recurrent Neural Networks

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class BidirectionalLSTMFromScratch:
    """
    Bi-Directional Sequence Modeling Engine.
    Processes sequences in two opposing directions concurrently:
      1. Forward Pass: Computes temporal representations conditioning on past history: h_forward_t
      2. Backward Pass: Computes temporal representations conditioning on future context: h_backward_t
      3. Fusion: Concatenates forward and backward states: h_t = [h_forward_t, h_backward_t]
    Enables rich contextual comprehension across sequence tagging, NER, and sentiment classification.
    """
    def __init__(self, input_dim=4, hidden_dim=8, random_state=42):
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Forward LSTM weights
        concat_dim = input_dim + hidden_dim
        self.W_fwd = np.random.randn(concat_dim, 4 * hidden_dim) * 0.1
        self.b_fwd = np.zeros((1, 4 * hidden_dim))
        self.b_fwd[0, :hidden_dim] = 1.0 # Forget gate bias = 1.0

        # Backward LSTM weights (independent parameter manifold)
        self.W_bwd = np.random.randn(concat_dim, 4 * hidden_dim) * 0.1
        self.b_bwd = np.zeros((1, 4 * hidden_dim))
        self.b_bwd[0, :hidden_dim] = 1.0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def _lstm_step(self, x_t, h_prev, c_prev, W, b):
        concat = np.hstack((x_t, h_prev))
        gates = np.dot(concat, W) + b
        D = self.hidden_dim

        f = self.sigmoid(gates[:, :D])
        i = self.sigmoid(gates[:, D:2*D])
        c_bar = np.tanh(gates[:, 2*D:3*D])
        o = self.sigmoid(gates[:, 3*D:])

        c_next = f * c_prev + i * c_bar
        h_next = o * np.tanh(c_next)
        return h_next, c_next

    def process_bidirectional(self, sequence_vectors):
        """
        sequence_vectors: numpy array of shape (T, input_dim)
        Returns: concatenated bidirectional hidden representations of shape (T, 2 * hidden_dim)
        """
        T = len(sequence_vectors)
        fwd_states = []
        bwd_states = [None] * T

        # 1. Forward Pass (t = 0 -> T-1)
        h_f = np.zeros((1, self.hidden_dim))
        c_f = np.zeros((1, self.hidden_dim))
        for t in range(T):
            x_t = sequence_vectors[t:t+1]
            h_f, c_f = self._lstm_step(x_t, h_f, c_f, self.W_fwd, self.b_fwd)
            fwd_states.append(h_f)

        # 2. Backward Pass (t = T-1 -> 0)
        h_b = np.zeros((1, self.hidden_dim))
        c_b = np.zeros((1, self.hidden_dim))
        for t in reversed(range(T)):
            x_t = sequence_vectors[t:t+1]
            h_b, c_b = self._lstm_step(x_t, h_b, c_b, self.W_bwd, self.b_bwd)
            bwd_states[t] = h_b

        # 3. Concatenate bidirectional representations at each step
        bi_representations = []
        for t in range(T):
            concat_h = np.hstack((fwd_states[t], bwd_states[t]))
            bi_representations.append(concat_h)

        return np.vstack(bi_representations)


if __name__ == "__main__":
    print("=" * 65)
    print("Week 10 Task 1: Bi-Directional LSTM Sequential Architecture")
    print("Student: Abubakar Iqbal | NAVTTC Week 10")
    print("=" * 65)

    input_dim = 4
    hidden_dim = 6
    bilstm = BidirectionalLSTMFromScratch(input_dim=input_dim, hidden_dim=hidden_dim, random_state=42)

    # Sequence of 5 word embedding representations
    # Simulating sentence: ["The", "bank", "approved", "my", "loan"]
    np.random.seed(42)
    sample_tokens = ["The", "bank", "approved", "my", "loan"]
    sequence = np.random.randn(len(sample_tokens), input_dim)

    print(f"\nInput Sequence of {len(sample_tokens)} Tokens: {sample_tokens}")
    print(f"Token Vector Dimension: {input_dim}")
    print(f"Forward Hidden State Dimension: {hidden_dim}")
    print(f"Backward Hidden State Dimension: {hidden_dim}")
    print(f"Fused Bi-Directional Dimension: {2 * hidden_dim}")

    bi_output = bilstm.process_bidirectional(sequence)

    print("\n--- Fused Bi-Directional Representations (T = 5, Dim = 12) ---")
    for idx, token in enumerate(sample_tokens):
        vec = bi_output[idx]
        fwd_part = vec[:hidden_dim].round(3)
        bwd_part = vec[hidden_dim:].round(3)
        print(f"Token '{token:<8}' -> Forward: {fwd_part} | Backward: {bwd_part}")

    print("\nWeek 10 Task 1 Bi-Directional LSTM Pipeline Verified.")
