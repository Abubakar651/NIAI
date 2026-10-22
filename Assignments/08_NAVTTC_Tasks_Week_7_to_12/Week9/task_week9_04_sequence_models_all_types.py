# Week 9 - Task 04: The 4 Canonical Sequence Modeling Paradigms
# NAVTTC Outline Week 9 Days 3, 4, 5
# Reference: Karpathy (2015) / The Unreasonable Effectiveness of Recurrent Neural Networks

import numpy as np
import pandas as pd

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class SequenceModelSuite:
    """
    Implements and demonstrates all 4 foundational sequence modeling archetypes:
      1. One-to-One (1-to-1): Fixed Input -> Fixed Output (Traditional Classifier)
      2. One-to-Many (1-to-M): Fixed Seed Vector -> Sequential Output (Text/Music Generation, Image Captioning)
      3. Many-to-One (M-to-1): Sequential Input -> Fixed Summary Output (Sentiment Analysis, Document Classification)
      4. Many-to-Many (M-to-M): Sequential Input -> Sequential Output (Seq2Seq Translation, POS Tagging)
    """

    @staticmethod
    def one_to_one(input_vec):
        """Standard Feedforward: Predicts default risk probability from single applicant vector."""
        W = np.array([[0.4], [0.8], [-0.5]])
        bias = 0.1
        logit = np.dot(input_vec, W) + bias
        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(prob[0, 0])

    @staticmethod
    def one_to_many(seed_vector, num_steps=5):
        """Generates a sequential trajectory from a single seed vector."""
        W_state = np.array([
            [0.8, -0.2],
            [0.1,  0.7]
        ])
        state = seed_vector.copy()
        generated_sequence = []
        for _ in range(num_steps):
            state = np.tanh(np.dot(state, W_state))
            generated_sequence.append(float(state[0, 0]))
        return generated_sequence

    @staticmethod
    def many_to_one(sequence_tokens):
        """Processes sequential text tokens and outputs a single sentiment/class prediction."""
        # Simple token polarity embedding lookup
        lexicon = {
            "fast": 0.8, "secure": 0.9, "reliable": 0.85, "love": 0.95,
            "slow": -0.8, "bug": -0.7, "crash": -0.9, "terrible": -0.95,
            "banking": 0.0, "app": 0.0, "service": 0.0
        }
        h = 0.0
        alpha = 0.6  # decay parameter
        for token in sequence_tokens:
            val = lexicon.get(token.lower(), 0.0)
            h = alpha * h + (1 - alpha) * val
        prob_positive = 1.0 / (1.0 + np.exp(-3.0 * h))
        return prob_positive

    @staticmethod
    def many_to_many_synced(sequence_tokens):
        """Synced Many-to-Many: Translates each token in sequence into a POS tag."""
        pos_rules = {
            "the": "DET", "bank": "NOUN", "approves": "VERB", "instant": "ADJ",
            "loans": "NOUN", "rapidly": "ADV", "and": "CONJ", "safely": "ADV"
        }
        return [(token, pos_rules.get(token.lower(), "NOUN")) for token in sequence_tokens]


if __name__ == "__main__":
    print("=" * 65)
    print("Week 9 Task 4: The 4 Canonical Sequence Modeling Architectures")
    print("Student: Abubakar Iqbal | NAVTTC Week 9")
    print("=" * 65)

    suite = SequenceModelSuite()

    # 1. One-to-One
    applicant_vec = np.array([[0.8, 0.4, 0.2]]) # [Debt Ratio, Missed Payments, Credit Score Scaled]
    p_default = suite.one_to_one(applicant_vec)
    print("\n[Pattern 1: One-to-One (1 -> 1)]")
    print("  Input:  Fixed 3-D Feature Vector ->", applicant_vec[0])
    print(f"  Output: Single Predicted Default Probability: {p_default:.4f}")

    # 2. One-to-Many
    seed = np.array([[1.0, 0.5]])
    trajectory = suite.one_to_many(seed, num_steps=5)
    print("\n[Pattern 2: One-to-Many (1 -> M)]")
    print("  Input:  Single Seed Initialization Vector ->", seed[0])
    print(f"  Output: Auto-Regressive Sequence of 5 Generated Values -> {[round(x, 4) for x in trajectory]}")

    # 3. Many-to-One
    review = ["the", "banking", "app", "is", "fast", "reliable", "and", "secure"]
    sentiment_p = suite.many_to_one(review)
    print("\n[Pattern 3: Many-to-One (M -> 1)]")
    print(f"  Input:  Temporal Sequence of {len(review)} Words -> \"{' '.join(review)}\"")
    print(f"  Output: Single Aggregated Sentiment Probability: {sentiment_p:.4f} (Positive: {sentiment_p > 0.5})")

    # 4. Many-to-Many
    sentence = ["the", "bank", "approves", "instant", "loans", "rapidly", "and", "safely"]
    tagged_seq = suite.many_to_many_synced(sentence)
    print("\n[Pattern 4: Many-to-Many (M -> M)]")
    print(f"  Input:  Sentence Sequence ({len(sentence)} Tokens)")
    print(f"  Output: Synchronous Token-Level POS Tag Sequence:")
    for word, tag in tagged_seq:
        print(f"    {word:<10} --> [{tag}]")

    print("\n" + "=" * 65)
    print("All 4 Canonical Sequence Model Architectures Demonstrated.")
    print("=" * 65)
