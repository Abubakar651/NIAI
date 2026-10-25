# Abubakar Iqbal
# NAVTTC Week 10 - Task 02: Scaled Dot-Product Attention

import numpy as np
import pandas as pd

class AttentionMechanism:
    def __init__(self, d_model=6):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)

    @staticmethod
    def softmax(scores):
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    def forward(self, Q, K, V):
        raw_scores = np.dot(Q, K.T) / self.scale
        weights = self.softmax(raw_scores)
        context = np.dot(weights, V)
        return context, weights

if __name__ == "__main__":
    print("NAVTTC Week 10 Task 02 - Abubakar Iqbal")

    abubakar_tokens = ["Abubakar", "submitted", "urgent", "credit", "application"]
    d_k = 6
    np.random.seed(42)

    embeddings = np.random.randn(len(abubakar_tokens), d_k)
    K = embeddings
    V = embeddings
    Q = embeddings[0:1]

    abubakar_attention = AttentionMechanism(d_model=d_k)
    context_vec, weights = abubakar_attention.forward(Q, K, V)

    print(f"Query Token: '{abubakar_tokens[0]}'")
    print(f"Computed Context Vector Shape: {context_vec.shape}")

    weights_df = pd.DataFrame({
        "Token": abubakar_tokens,
        "Attention Weight": weights[0].round(4)
    })
    print("\nAttention Distribution:")
    print(weights_df.to_string(index=False))
