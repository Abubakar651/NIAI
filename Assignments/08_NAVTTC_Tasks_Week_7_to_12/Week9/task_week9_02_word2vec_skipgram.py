# Week 9 - Task 02: Word2Vec Continuous Skip-Gram from Scratch
# NAVTTC Outline Week 9 Day 1
# Reference: Distributed Representations of Words and Phrases (Mikolov et al., 2013)

import numpy as np

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class Word2VecSkipGram:
    """
    Continuous Skip-Gram Model implemented from first principles.
    Inverts the CBOW objective: Takes a single center target word w_t
    and predicts the probability distribution of context words.
    """
    def __init__(self, vocab_size, embedding_dim=10, learning_rate=0.05, random_state=42):
        np.random.seed(random_state)
        self.V = vocab_size
        self.D = embedding_dim
        self.lr = learning_rate

        # Target center word matrix (W_in) and Context word matrix (W_out)
        self.W_in = np.random.randn(self.V, self.D) * 0.1
        self.W_out = np.random.randn(self.D, self.V) * 0.1

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def forward(self, target_idx):
        """Fetches embedding of target word and projects to vocabulary distribution."""
        v_target = self.W_in[target_idx:target_idx+1, :] # shape (1, D)
        u = np.dot(v_target, self.W_out) # shape (1, V)
        y_prob = self.softmax(u)
        return v_target, y_prob

    def train_step(self, target_idx, context_indices):
        """
        Updates weights based on multi-target cross-entropy loss across all
        context words within the window.
        """
        v_target, y_prob = self.forward(target_idx)
        total_loss = 0.0

        # Accumulated gradient on projection
        e_sum = np.zeros((1, self.V))
        for ctx_idx in context_indices:
            total_loss -= np.log(y_prob[0, ctx_idx] + 1e-12)
            e = np.copy(y_prob)
            e[0, ctx_idx] -= 1.0
            e_sum += e

        # Backpropagation
        dW_out = np.dot(v_target.T, e_sum) # (D x V)
        dv_target = np.dot(e_sum, self.W_out.T) # (1 x D)

        # SGD updates
        self.W_out -= self.lr * dW_out
        self.W_in[target_idx, :] -= self.lr * dv_target.flatten()

        return total_loss / len(context_indices)

    def get_embedding(self, word_idx):
        return self.W_in[word_idx, :]


if __name__ == "__main__":
    print("=" * 65)
    print("Week 9 Task 2: Word2Vec Continuous Skip-Gram Architecture")
    print("Student: Abubakar Iqbal | NAVTTC Week 9")
    print("=" * 65)

    corpus = "credit risk prediction requires historical financial data loan evaluation default analysis"
    tokens = corpus.lower().split()
    vocab = sorted(list(set(tokens)))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    V = len(vocab)

    print(f"Corpus: \"{corpus}\"")
    print(f"Vocabulary Size: {V} Unique Terms")

    # Generate Skip-Gram training pairs (target -> [contexts])
    window_size = 2
    skipgram_data = []
    for i in range(window_size, len(tokens) - window_size):
        target = word2idx[tokens[i]]
        contexts = [word2idx[tokens[j]] for j in range(i - window_size, i + window_size + 1) if j != i]
        skipgram_data.append((target, contexts))

    print(f"\nGenerated {len(skipgram_data)} Skip-Gram Target-to-Context tuples.")
    sample_tgt, sample_ctx = skipgram_data[0]
    print(f"Sample: Target '{idx2word[sample_tgt]}' -> Context {[idx2word[c] for c in sample_ctx]}")

    skipgram = Word2VecSkipGram(vocab_size=V, embedding_dim=8, learning_rate=0.08)

    print("\nTraining Skip-Gram Dense Embeddings...")
    for epoch in range(1, 201):
        losses = [skipgram.train_step(tgt, ctx) for tgt, ctx in skipgram_data]
        if epoch == 1 or epoch % 50 == 0 or epoch == 200:
            print(f"  Epoch {epoch:3d}/200 | Average Cross-Entropy Loss: {np.mean(losses):.5f}")

    print("\nTrained Word Embeddings (Skip-Gram Representations):")
    for word in ["credit", "risk", "loan", "default"]:
        emb = skipgram.get_embedding(word2idx[word])
        print(f"  Embedding('{word}'): {emb.round(3)}")

    print("\nWeek 9 Task 2 Continuous Skip-Gram Model Verified.")
