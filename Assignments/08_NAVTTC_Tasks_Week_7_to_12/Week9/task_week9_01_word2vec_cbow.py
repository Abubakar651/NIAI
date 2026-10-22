# Week 9 - Task 01: Word2Vec Continuous Bag-of-Words (CBOW) from Scratch
# NAVTTC Outline Week 9 Day 1
# Reference: Mikolov et al. (2013) / Efficient Estimation of Word Representations in Vector Space

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class Word2VecCBOW:
    """
    Continuous Bag-of-Words (CBOW) Word2Vec Model implemented from scratch with NumPy.
    Objective: Predict a central target word w_t given its surrounding context words [w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}].
    Features:
      - Projection/Embedding Matrix W1: (V x D)
      - Context-to-Hidden Average Pooling
      - Output Softmax Matrix W2: (D x V)
      - Backpropagation and SGD Embedding Updates
    """
    def __init__(self, vocab_size, embedding_dim=10, learning_rate=0.05, random_state=42):
        np.random.seed(random_state)
        self.V = vocab_size
        self.D = embedding_dim
        self.lr = learning_rate

        # Initialize input embedding matrix (W1) and output context matrix (W2)
        self.W1 = np.random.randn(self.V, self.D) * 0.1
        self.W2 = np.random.randn(self.D, self.V) * 0.1

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def forward(self, context_indices):
        """
        context_indices: list of token IDs in the context window.
        Averages the embedding vectors of context words.
        """
        # Average input embeddings
        h = np.mean(self.W1[context_indices, :], axis=0, keepdims=True) # shape (1, D)
        u = np.dot(h, self.W2) # shape (1, V)
        y_pred = self.softmax(u)
        return h, y_pred

    def train_step(self, context_indices, target_index):
        # 1. Forward Pass
        h, y_pred = self.forward(context_indices)

        # 2. Loss computation (Cross Entropy)
        loss = -np.log(y_pred[0, target_index] + 1e-12)

        # 3. Output layer gradient
        e = np.copy(y_pred)
        e[0, target_index] -= 1.0 # dL/du = y_pred - y_true

        # 4. Hidden layer gradients
        dW2 = np.dot(h.T, e) # (D x 1) x (1 x V) = (D x V)
        dh = np.dot(e, self.W2.T) # (1 x V) x (V x D) = (1 x D)

        # 5. Parameter updates
        self.W2 -= self.lr * dW2
        C = len(context_indices)
        for idx in context_indices:
            self.W1[idx, :] -= (self.lr / C) * dh.flatten()

        return loss

    def get_word_embedding(self, word_idx):
        return self.W1[word_idx, :]


if __name__ == "__main__":
    print("=" * 65)
    print("Week 9 Task 1: Word2Vec Continuous Bag-of-Words (CBOW) Engine")
    print("Student: Abubakar Iqbal | NAVTTC Week 9")
    print("=" * 65)

    corpus = "artificial intelligence machine learning deep learning neural network model training predictive banking data analysis"
    tokens = corpus.lower().split()
    vocab = sorted(list(set(tokens)))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    V = len(vocab)

    print(f"Corpus Length: {len(tokens)} Tokens | Unique Vocabulary Size: {V}")
    print("Vocabulary Index Mapping:", {k: word2idx[k] for k in vocab[:6]})

    # Generate CBOW training samples (Window size = 2)
    # [context_words] -> target_word
    window_size = 2
    cbow_data = []
    for i in range(window_size, len(tokens) - window_size):
        context = [word2idx[tokens[j]] for j in range(i - window_size, i + window_size + 1) if j != i]
        target = word2idx[tokens[i]]
        cbow_data.append((context, target))

    print(f"\nGenerated {len(cbow_data)} (Context, Target) Training Pairs.")
    sample_ctx, sample_tgt = cbow_data[0]
    print(f"Sample Pair: Context {[idx2word[c] for c in sample_ctx]} -> Target '{idx2word[sample_tgt]}'")

    # Instantiate CBOW with 8-dimensional dense embeddings
    cbow = Word2VecCBOW(vocab_size=V, embedding_dim=8, learning_rate=0.08)

    print("\nTraining CBOW Projection Matrix via Backpropagation...")
    for epoch in range(1, 201):
        losses = [cbow.train_step(ctx, tgt) for ctx, tgt in cbow_data]
        if epoch == 1 or epoch % 50 == 0 or epoch == 200:
            print(f"  Epoch {epoch:3d}/200 | Average Cross-Entropy Loss: {np.mean(losses):.5f}")

    print("\nLearned 8-Dimensional Dense Word Vectors:")
    for w in ["intelligence", "learning", "neural", "banking"]:
        emb = cbow.get_word_embedding(word2idx[w])
        print(f"  Vector('{w}'): {emb.round(3)}")

    print("\nWeek 9 Task 1 CBOW Word2Vec Training Completed.")
