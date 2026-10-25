# Abubakar Iqbal
# NAVTTC Week 9 - Task 04: The 4 Sequence Modeling Paradigms

import numpy as np
import pandas as pd

class SequenceModels:
    @staticmethod
    def one_to_one(input_vec):
        W = np.array([[0.4], [0.8], [-0.5]])
        bias = 0.1
        logit = np.dot(input_vec, W) + bias
        prob = 1.0 / (1.0 + np.exp(-logit))
        return float(prob[0, 0])

    @staticmethod
    def one_to_many(seed, steps=5):
        W_state = np.array([[0.8, -0.2], [0.1, 0.7]])
        state = seed.copy()
        out = []
        for _ in range(steps):
            state = np.tanh(np.dot(state, W_state))
            out.append(float(state[0, 0]))
        return out

    @staticmethod
    def many_to_one(tokens):
        lexicon = {
            "fast": 0.8, "secure": 0.9, "reliable": 0.85, "love": 0.95,
            "slow": -0.8, "bug": -0.7, "crash": -0.9, "terrible": -0.95
        }
        h = 0.0
        alpha = 0.6
        for t in tokens:
            val = lexicon.get(t.lower(), 0.0)
            h = alpha * h + (1 - alpha) * val
        return 1.0 / (1.0 + np.exp(-3.0 * h))

    @staticmethod
    def many_to_many(tokens):
        pos_dict = {
            "abubakar": "PROPN",
            "iqbal": "PROPN",
            "studies": "VERB",
            "deep": "ADJ",
            "learning": "NOUN",
            "models": "NOUN",
            "in": "ADP",
            "lahore": "PROPN"
        }
        return [(t, pos_dict.get(t.lower(), "NOUN")) for t in tokens]

if __name__ == "__main__":
    print("NAVTTC Week 9 Task 04 - Abubakar Iqbal")

    abubakar_applicant = np.array([[0.85, 0.70, 0.20]])
    risk_prob = SequenceModels.one_to_one(abubakar_applicant)
    print(f"1-to-1 (Single Input -> Single Output): Risk Prob = {risk_prob:.4f}")

    abubakar_seed = np.array([[0.5, 0.5]])
    trajectory = SequenceModels.one_to_many(abubakar_seed, steps=4)
    print(f"1-to-Many (Seed Vector -> Sequence): {np.round(trajectory, 3)}")

    abubakar_review = ["abubakar", "found", "the", "system", "fast", "reliable", "and", "secure"]
    sentiment_score = SequenceModels.many_to_one(abubakar_review)
    print(f"Many-to-1 (Sequence -> Single Output): Positive Prob = {sentiment_score:.4f}")

    abubakar_sentence = ["Abubakar", "studies", "deep", "learning", "in", "Lahore"]
    tagged_seq = SequenceModels.many_to_many(abubakar_sentence)
    print("\nMany-to-Many (Sequence -> Sequence Tagging):")
    for token, tag in tagged_seq:
        print(f"  {token:<10} -> {tag}")
