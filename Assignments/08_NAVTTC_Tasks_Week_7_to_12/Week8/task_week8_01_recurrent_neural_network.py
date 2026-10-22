# Week 8 - Task 01: Recurrent Neural Networks (RNN) from Scratch
# NAVTTC Outline Week 8 Day 1
# Reference: Deep Learning with Sequential Data / RNN Architecture

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class SimpleRNNFromScratch:
    """
    Recurrent Neural Network (RNN) implemented from foundational mathematics with NumPy.
    Models sequential temporal dependencies via a recurrent hidden state.
    Equations:
      h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
      y_t = softmax(W_hy * h_t + b_y)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # Weight matrices initialized with Xavier normal scaling
        np.random.seed(42)
        self.W_xh = np.random.randn(input_dim, hidden_dim) * 0.1
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_hy = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b_h = np.zeros((1, hidden_dim))
        self.b_y = np.zeros((1, output_dim))

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, inputs):
        """
        inputs: List of shape (1, input_dim) representing a temporal sequence [x_0, x_1, ..., x_T]
        """
        T = len(inputs)
        h_states = {}
        y_preds = {}
        h_states[-1] = np.zeros((1, self.hidden_dim))

        for t in range(T):
            x_t = inputs[t]
            # Recurrent hidden activation
            z_h = np.dot(x_t, self.W_xh) + np.dot(h_states[t - 1], self.W_hh) + self.b_h
            h_states[t] = np.tanh(z_h)
            # Output projection
            z_y = np.dot(h_states[t], self.W_hy) + self.b_y
            y_preds[t] = self.softmax(z_y)

        return y_preds, h_states

    def train_step(self, inputs, targets):
        """
        Backpropagation Through Time (BPTT) implementation.
        """
        T = len(inputs)
        y_preds, h_states = self.forward(inputs)

        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((1, self.hidden_dim))
        total_loss = 0.0

        for t in reversed(range(T)):
            target = targets[t]
            prob = y_preds[t]
            # Cross-entropy loss
            total_loss -= np.log(prob[0, target] + 1e-12)

            # Gradient on output
            dy = np.copy(prob)
            dy[0, target] -= 1.0  # dL/dy for softmax + cross entropy

            dW_hy += np.dot(h_states[t].T, dy)
            db_y += dy

            # Gradient backpropagated into hidden state
            dh = np.dot(dy, self.W_hy.T) + dh_next
            # Derivative of tanh: 1 - h^2
            dh_raw = (1.0 - h_states[t] ** 2) * dh

            db_h += dh_raw
            dW_xh += np.dot(inputs[t].T, dh_raw)
            dW_hh += np.dot(h_states[t - 1].T, dh_raw)
            dh_next = np.dot(dh_raw, self.W_hh.T)

        # Gradient clipping to prevent exploding gradients
        for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(grad, -5.0, 5.0, out=grad)

        # SGD parameter updates
        self.W_xh -= self.lr * dW_xh
        self.W_hh -= self.lr * dW_hh
        self.W_hy -= self.lr * dW_hy
        self.b_h -= self.lr * db_h
        self.b_y -= self.lr * db_y

        return total_loss / T


if __name__ == "__main__":
    print("=" * 65)
    print("Week 8 Task 1: Recurrent Neural Network (RNN & BPTT) from Scratch")
    print("Student: Abubakar Iqbal | NAVTTC Week 8")
    print("=" * 65)

    # Sequence Learning Problem: Learning character sequences
    text = "deep learning with rnn"
    chars = sorted(list(set(text)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    print(f"Target Text Sequence: \"{text}\"")
    print(f"Vocabulary Size: {vocab_size} Characters ({chars})")

    # Encode input characters as One-Hot Vectors
    def one_hot(char_idx):
        vec = np.zeros((1, vocab_size))
        vec[0, char_idx] = 1.0
        return vec

    inputs_seq = [one_hot(char_to_ix[ch]) for ch in text[:-1]]
    targets_seq = [char_to_ix[ch] for ch in text[1:]]

    rnn = SimpleRNNFromScratch(input_dim=vocab_size, hidden_dim=32, output_dim=vocab_size, learning_rate=0.08)

    print("\nTraining RNN via Backpropagation Through Time (BPTT)...")
    for epoch in range(1, 301):
        loss = rnn.train_step(inputs_seq, targets_seq)
        if epoch == 1 or epoch % 50 == 0 or epoch == 300:
            print(f"Epoch {epoch:3d}/300 | Average Sequence Loss: {loss:.5f}")

    # Sequence Generation / Prediction
    y_preds, _ = rnn.forward(inputs_seq)
    predicted_chars = [ix_to_char[np.argmax(y_preds[t])] for t in range(len(inputs_seq))]
    print(f"\nOriginal Target: \"{text[1:]}\"")
    print(f"RNN Prediction:  \"{''.join(predicted_chars)}\"")
    print("Week 8 Task 1 RNN Model Completed Successfully.")
