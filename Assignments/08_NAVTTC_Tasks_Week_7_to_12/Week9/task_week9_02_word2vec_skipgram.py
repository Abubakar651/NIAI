# Abubakar Iqbal
# NAVTTC Week 9 - Task 02: Word2Vec Skip-Gram from Scratch

import numpy as np

class Word2VecSkipGram:
    def __init__(self, vocab_size, emb_dim=8, lr=0.08, seed=42):
        np.random.seed(seed)
        self.V = vocab_size
        self.D = emb_dim
        self.lr = lr
        self.W_in = np.random.randn(self.V, self.D) * 0.1
        self.W_out = np.random.randn(self.D, self.V) * 0.1

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def forward(self, target_idx):
        v = self.W_in[target_idx:target_idx+1, :]
        u = np.dot(v, self.W_out)
        return v, self.softmax(u)

    def train_step(self, target_idx, context_indices):
        v, y_prob = self.forward(target_idx)
        loss = 0.0
        e_sum = np.zeros((1, self.V))

        for c_idx in context_indices:
            loss -= np.log(y_prob[0, c_idx] + 1e-12)
            e = np.copy(y_prob)
            e[0, c_idx] -= 1.0
            e_sum += e

        dW_out = np.dot(v.T, e_sum)
        dv = np.dot(e_sum, self.W_out.T)

        self.W_out -= self.lr * dW_out
        self.W_in[target_idx, :] -= self.lr * dv.flatten()
        return loss

    def get_embedding(self, idx):
        return self.W_in[idx, :]

if __name__ == "__main__":
    print("NAVTTC Week 9 Task 02 - Abubakar Iqbal")

    abubakar_corpus = (
        "abubakar explores machine learning deep learning word representations "
        "abubakar develops skipgram neural models for natural language processing"
    )
    tokens = abubakar_corpus.lower().split()
    vocab = sorted(list(set(tokens)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}

    window_size = 2
    skipgram_data = []
    for i in range(window_size, len(tokens) - window_size):
        tgt = w2i[tokens[i]]
        ctx = [w2i[tokens[j]] for j in range(i - window_size, i + window_size + 1) if j != i]
        skipgram_data.append((tgt, ctx))

    abubakar_skipgram = Word2VecSkipGram(vocab_size=len(vocab), emb_dim=6, lr=0.1)

    for epoch in range(1, 151):
        losses = [abubakar_skipgram.train_step(tgt, ctx) for tgt, ctx in skipgram_data]
        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Mean Loss: {np.mean(losses):.4f}")

    print("\nLearned Skip-Gram Vectors:")
    for word in ["abubakar", "learning", "neural", "models"]:
        vec = abubakar_skipgram.get_embedding(w2i[word])
        print(f"Vector('{word}'): {vec.round(3)}")
