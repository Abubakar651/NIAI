# Abubakar Iqbal
# NAVTTC Week 9 - Task 01: Word2Vec CBOW from Scratch

import numpy as np

class Word2VecCBOW:
    def __init__(self, vocab_size, emb_dim=8, lr=0.08, seed=42):
        np.random.seed(seed)
        self.V = vocab_size
        self.D = emb_dim
        self.lr = lr
        self.W1 = np.random.randn(self.V, self.D) * 0.1
        self.W2 = np.random.randn(self.D, self.V) * 0.1

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def forward(self, context_indices):
        h = np.mean(self.W1[context_indices, :], axis=0, keepdims=True)
        u = np.dot(h, self.W2)
        y_pred = self.softmax(u)
        return h, y_pred

    def train_step(self, context_indices, target_index):
        h, y_pred = self.forward(context_indices)
        loss = -np.log(y_pred[0, target_index] + 1e-12)

        e = np.copy(y_pred)
        e[0, target_index] -= 1.0

        dW2 = np.dot(h.T, e)
        dh = np.dot(e, self.W2.T)

        self.W2 -= self.lr * dW2
        C = len(context_indices)
        for idx in context_indices:
            self.W1[idx, :] -= (self.lr / C) * dh.flatten()
        return loss

    def get_embedding(self, idx):
        return self.W1[idx, :]

if __name__ == "__main__":
    print("NAVTTC Week 9 Task 01 - Abubakar Iqbal")

    abubakar_corpus = (
        "abubakar studies artificial intelligence and deep learning "
        "abubakar trains neural networks for predictive models and data analysis"
    )
    tokens = abubakar_corpus.lower().split()
    vocab = sorted(list(set(tokens)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}

    window_size = 2
    cbow_data = []
    for i in range(window_size, len(tokens) - window_size):
        ctx = [w2i[tokens[j]] for j in range(i - window_size, i + window_size + 1) if j != i]
        tgt = w2i[tokens[i]]
        cbow_data.append((ctx, tgt))

    abubakar_cbow = Word2VecCBOW(vocab_size=len(vocab), emb_dim=6, lr=0.1)

    for epoch in range(1, 151):
        losses = [abubakar_cbow.train_step(ctx, tgt) for ctx, tgt in cbow_data]
        if epoch == 1 or epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Mean Loss: {np.mean(losses):.4f}")

    print("\nLearned Embeddings Sample:")
    for word in ["abubakar", "intelligence", "neural", "models"]:
        emb = abubakar_cbow.get_embedding(w2i[word])
        print(f"Vector('{word}'): {emb.round(3)}")
