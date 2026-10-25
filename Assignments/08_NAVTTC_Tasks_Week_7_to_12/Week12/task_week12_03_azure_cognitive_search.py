# Abubakar Iqbal
# NAVTTC Week 12 - Task 03: Knowledge Mining with Azure AI Search

import math
import re

class AzureCognitiveSearchIndex:
    def __init__(self):
        self.documents = []
        self.inverted_index = {}
        self.doc_lengths = {}
        self.avg_len = 0.0

    def add_document(self, doc_id, title, content):
        tokens = [w.lower() for w in re.findall(r"\w+", content)]
        doc = {
            "id": doc_id,
            "title": title,
            "content": content,
            "tokens": tokens,
            "length": len(tokens)
        }
        self.documents.append(doc)
        self.doc_lengths[doc_id] = len(tokens)
        self.avg_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        for t in set(tokens):
            if t not in self.inverted_index:
                self.inverted_index[t] = []
            self.inverted_index[t].append(doc_id)

    def search(self, query, top_k=3):
        q_tokens = [w.lower() for w in re.findall(r"\w+", query)]
        scores = {}
        N = len(self.documents)
        k1 = 1.5
        b = 0.75

        for q in q_tokens:
            if q not in self.inverted_index:
                continue
            doc_ids = self.inverted_index[q]
            df = len(doc_ids)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for doc in self.documents:
                d_id = doc["id"]
                if d_id not in doc_ids:
                    continue
                tf = doc["tokens"].count(q)
                denom = tf + k1 * (1 - b + b * (doc["length"] / (self.avg_len + 1e-6)))
                score = idf * (tf * (k1 + 1)) / denom
                scores[d_id] = scores.get(d_id, 0.0) + score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for d_id, sc in sorted_docs:
            doc_obj = next(d for d in self.documents if d["id"] == d_id)
            results.append((doc_obj["title"], round(sc, 4), doc_obj["content"]))
        return results

if __name__ == "__main__":
    print("NAVTTC Week 12 Task 03 - Abubakar Iqbal")

    abubakar_search = AzureCognitiveSearchIndex()

    abubakar_search.add_document(
        "DOC-1",
        "Abubakar Iqbal Lending Framework",
        "Abubakar Iqbal implemented automated risk scoring and credit threshold policies for digital banking."
    )
    abubakar_search.add_document(
        "DOC-2",
        "NETSOL Machine Learning Guidelines",
        "Standard guidelines for deploying neural network models, monitoring concept drift, and validation."
    )
    abubakar_search.add_document(
        "DOC-3",
        "State Bank Regulatory Compliance",
        "Mandatory customer verification, Anti-Money Laundering AML checks, and financial reporting rules."
    )

    query = "Abubakar credit scoring risk"
    print(f"\nSearch Query: '{query}'")
    results = abubakar_search.search(query)

    print("\nRanked Knowledge Mining Results:")
    for rank, (title, score, content) in enumerate(results, 1):
        print(f"Rank {rank} (BM25: {score}) -> {title}")
        print(f"  Snippet: {content}")
