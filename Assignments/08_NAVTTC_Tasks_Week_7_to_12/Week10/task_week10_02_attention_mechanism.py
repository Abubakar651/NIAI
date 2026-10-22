# Week 10 - Task 02: Attention Mechanism in Deep Learning Sequence Models
# NAVTTC Outline Week 10 Days 2 & 3
# Reference: Vaswani et al. (2017) / Attention Is All You Need; Bahdanau et al. (2014)

import numpy as np
import pandas as pd

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class ScaledDotProductAttention:
    """
    Mathematical Implementation of the Scaled Dot-Product Attention Mechanism.
    Formula:
      Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
    Where:
      Q (Query): What we are looking for (e.g. current decoder state or query token)
      K (Key): What context tokens are available to be matched against
      V (Value): The actual informational representation to retrieve
      d_k: Scaling factor to prevent vanishing gradients in the softmax region
    """
    def __init__(self, d_model=8):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)

    @staticmethod
    def softmax(scores, axis=-1):
        exp_s = np.exp(scores - np.max(scores, axis=axis, keepdims=True))
        return exp_s / np.sum(exp_s, axis=axis, keepdims=True)

    def compute_attention(self, Q, K, V):
        """
        Q: shape (seq_len_q, d_k)
        K: shape (seq_len_k, d_k)
        V: shape (seq_len_k, d_v)
        """
        # 1. Raw compatibility dot product: (seq_len_q x seq_len_k)
        raw_scores = np.dot(Q, K.T)

        # 2. Scaling by sqrt(d_k)
        scaled_scores = raw_scores / self.scale

        # 3. Softmax to obtain attention distribution weights (summing to 1 across keys)
        attention_weights = self.softmax(scaled_scores, axis=-1)

        # 4. Context Vector: Weighted sum of Value representations
        context_vector = np.dot(attention_weights, V)

        return context_vector, attention_weights


if __name__ == "__main__":
    print("=" * 65)
    print("Week 10 Task 2: Scaled Dot-Product Attention Mechanism")
    print("Student: Abubakar Iqbal | NAVTTC Week 10")
    print("=" * 65)

    d_k = 6 # Embedding / projection dimension
    attention = ScaledDotProductAttention(d_model=d_k)

    # Context sequence: ["Customer", "requested", "urgent", "loan", "approval"]
    tokens = ["Customer", "requested", "urgent", "loan", "approval"]
    seq_len = len(tokens)

    np.random.seed(42)
    # Simulate Query, Key, and Value matrices
    # For Self-Attention: Q, K, V are linear projections of input embeddings
    X = np.random.randn(seq_len, d_k)
    W_Q = np.random.randn(d_k, d_k) * 0.5
    W_K = np.random.randn(d_k, d_k) * 0.5
    W_V = np.random.randn(d_k, d_k) * 0.5

    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)

    print(f"Context Sequence: {tokens}")
    print(f"Key / Query Feature Dimension (d_k): {d_k}")
    print(f"Scaling Factor sqrt(d_k): {np.sqrt(d_k):.4f}")

    context, weights = attention.compute_attention(Q, K, V)

    print("\n--- Attention Weights Matrix (Softmax Alignment Heatmap) ---")
    df_weights = pd.DataFrame(weights, index=tokens, columns=tokens)
    print(df_weights.round(4))

    print("\nRow Sum Verification (Must equal 1.0 per query token):")
    for tok, row_sum in zip(tokens, np.sum(weights, axis=1)):
        print(f"  Token '{tok:<10}' Softmax Sum = {row_sum:.6f}")

    print("\nContext Vector Representation Shape:", context.shape)
    print(f"Sample Context Vector for '{tokens[3]}' (loan):\n  {context[3].round(3)}")

    print("\nWeek 10 Task 2 Attention Mechanism Executed Successfully.")
