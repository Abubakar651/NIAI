# Abubakar Iqbal
# NAVTTC Week 9 - Task 03: Custom Word Embedding Space and Vector Arithmetic

import numpy as np

class WordVectorSpace:
    def __init__(self):
        self.vectors = {
            "king":       np.array([ 0.90,  0.85, -0.10, -0.20,  0.80]),
            "queen":      np.array([ 0.92, -0.85, -0.10, -0.20,  0.80]),
            "man":        np.array([ 0.05,  0.90, -0.05, -0.05,  0.20]),
            "woman":      np.array([ 0.05, -0.90, -0.05, -0.05,  0.20]),
            "abubakar":   np.array([ 0.05,  0.75,  0.30,  0.92,  0.60]),
            "developer":  np.array([ 0.00,  0.50,  0.25,  0.90,  0.55]),
            "engineer":   np.array([ 0.00,  0.40,  0.35,  0.95,  0.65]),
            "bank":       np.array([-0.10,  0.00,  0.95,  0.15,  0.85]),
            "loan":       np.array([-0.05,  0.00,  0.90,  0.10,  0.60]),
            "borrower":   np.array([ 0.00,  0.10,  0.80,  0.05,  0.30]),
            "default":    np.array([-0.10,  0.00,  0.85, -0.10,  0.40]),
            "python":     np.array([-0.05,  0.00,  0.10,  0.95,  0.50]),
            "neural":     np.array([-0.05,  0.00,  0.15,  0.90,  0.55])
        }

    @staticmethod
    def cosine_similarity(u, v):
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    def nearest_neighbors(self, query_vec, top_k=3, exclude=None):
        exclude = exclude or []
        scores = []
        for word, vec in self.vectors.items():
            if word in exclude:
                continue
            sim = self.cosine_similarity(query_vec, vec)
            scores.append((word, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

if __name__ == "__main__":
    print("NAVTTC Week 9 Task 03 - Abubakar Iqbal")

    abubakar_space = WordVectorSpace()

    print("Cosine Similarities:")
    pairs = [("king", "queen"), ("abubakar", "developer"), ("bank", "loan"), ("python", "neural"), ("king", "python")]
    for w1, w2 in pairs:
        sim = abubakar_space.cosine_similarity(abubakar_space.vectors[w1], abubakar_space.vectors[w2])
        print(f"Similarity({w1}, {w2}) = {sim:.4f}")

    print("\nVector Arithmetic (King - Man + Woman):")
    v_analogy = (
        abubakar_space.vectors["king"] -
        abubakar_space.vectors["man"] +
        abubakar_space.vectors["woman"]
    )
    neighbors = abubakar_space.nearest_neighbors(v_analogy, top_k=3, exclude=["king", "man"])
    for word, sim in neighbors:
        print(f"Result candidate: '{word}' (Cosine: {sim:.4f})")

    print("\nNearest Neighbors for 'abubakar':")
    abubakar_matches = abubakar_space.nearest_neighbors(abubakar_space.vectors["abubakar"], top_k=3, exclude=["abubakar"])
    for word, sim in abubakar_matches:
        print(f"Match: '{word}' (Cosine: {sim:.4f})")
