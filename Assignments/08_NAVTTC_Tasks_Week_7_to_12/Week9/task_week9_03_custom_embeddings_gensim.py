# Week 9 - Task 03: Custom Word Embedding Training & Vector Arithmetic
# NAVTTC Outline Week 9 Day 2
# Reference: Word Vector Algebra & Semantic Cosine Geometry

import numpy as np
import pandas as pd

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class WordVectorSpace:
    """
    Semantic Vector Space Engine.
    Demonstrates:
      1. Cosine similarity between dense representation vectors.
      2. Word vector arithmetic (e.g. King - Man + Woman = Queen).
      3. Semantic nearest-neighbor clustering in embedding geometry.
    """
    def __init__(self):
        # Curated high-dimensional semantic vector projections (representing trained embeddings)
        # Dimensions represent latent semantic axes: [Royalty, Gender/Masculinity, Finance/Credit, Technology, Scale]
        self.vectors = {
            "king":       np.array([ 0.90,  0.85, -0.10, -0.20,  0.80]),
            "queen":      np.array([ 0.92, -0.85, -0.10, -0.20,  0.80]),
            "man":        np.array([ 0.05,  0.90, -0.05, -0.05,  0.20]),
            "woman":      np.array([ 0.05, -0.90, -0.05, -0.05,  0.20]),
            "bank":       np.array([-0.10,  0.00,  0.95,  0.15,  0.85]),
            "loan":       np.array([-0.05,  0.00,  0.90,  0.10,  0.60]),
            "borrower":   np.array([ 0.00,  0.10,  0.80,  0.05,  0.30]),
            "default":    np.array([-0.10,  0.00,  0.85, -0.10,  0.40]),
            "python":     np.array([-0.05,  0.00,  0.10,  0.95,  0.50]),
            "algorithm":  np.array([-0.05,  0.00,  0.20,  0.92,  0.60]),
            "neural":     np.array([-0.05,  0.00,  0.15,  0.90,  0.55]),
            "data":       np.array([-0.05,  0.00,  0.30,  0.88,  0.70])
        }

    @staticmethod
    def cosine_similarity(u, v):
        """Calculates cosine angle between two semantic vectors."""
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    def find_nearest_neighbors(self, query_vec, top_k=3, exclude=None):
        """Finds most semantically similar terms to query_vec in the embedding manifold."""
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
    print("=" * 65)
    print("Week 9 Task 3: Custom Word Embedding & Semantic Vector Geometry")
    print("Student: Abubakar Iqbal | NAVTTC Week 9")
    print("=" * 65)

    space = WordVectorSpace()

    # 1. Pairwise Cosine Similarities
    pairs = [
        ("bank", "loan"),
        ("bank", "default"),
        ("python", "algorithm"),
        ("algorithm", "neural"),
        ("king", "queen"),
        ("bank", "python") # Unrelated cross-domain pair
    ]

    print("\n[1] Pairwise Semantic Cosine Similarities:")
    for w1, w2 in pairs:
        sim = space.cosine_similarity(space.vectors[w1], space.vectors[w2])
        print(f"  Similarity('{w1}', '{w2}'): {sim:.4f}")

    # 2. Classic Semantic Analogy: Vector Arithmetic
    # Target: King - Man + Woman = ?
    print("\n[2] Executing Vector Algebra Analogy:")
    print("  Equation: Vector('king') - Vector('man') + Vector('woman')")
    analogy_vector = space.vectors["king"] - space.vectors["man"] + space.vectors["woman"]

    top_analogy = space.find_nearest_neighbors(analogy_vector, top_k=3, exclude=["king", "man", "woman"])
    print(f"  --> Top Predicted Match: '{top_analogy[0][0]}' (Cosine Similarity: {top_analogy[0][1]:.4f})")
    print(f"  Full Ranking: {top_analogy}")

    # 3. Domain Analogy: Algorithm - Python + Bank = ?
    print("\n[3] Domain Vector Shift:")
    print("  Equation: Vector('loan') - Vector('bank') + Vector('python')")
    domain_shift = space.vectors["loan"] - space.vectors["bank"] + space.vectors["python"]
    top_domain = space.find_nearest_neighbors(domain_shift, top_k=3, exclude=["loan", "bank", "python"])
    print(f"  --> Top Related Term: '{top_domain[0][0]}' (Cosine Similarity: {top_domain[0][1]:.4f})")

    print("\nWeek 9 Task 3 Vector Space Algebra Completed Successfully.")
