# Week 12 - Task 03: Knowledge Mining with Azure AI Search (Cognitive Search)
# NAVTTC Outline Week 12 Day 4
# Reference: Microsoft Learn / Implement Knowledge Mining with Azure AI Search

import math
import re
import pandas as pd

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class AzureCognitiveSearchIndex:
    """
    Simulation of an Azure AI Search (Cognitive Search) Knowledge Mining Solution.
    Architecture:
      1. Document Cracking: Ingesting raw unstructured PDFs and financial records.
      2. Cognitive Enrichment Skillset:
         - Optical Character Recognition (OCR) Skill
         - Entity Recognition Skill (Organizations, Locations, Amounts)
         - Key Phrase Extraction Skill
      3. Inverted Search Index: BM25 / TF-IDF scoring with semantic ranking.
    """
    def __init__(self):
        self.documents = []
        self.inverted_index = {}
        self.doc_lengths = {}
        self.avg_doc_len = 0.0

    def add_document_with_cognitive_skills(self, doc_id, title, raw_content):
        """Simulates ingestion through an Azure Cognitive Enrichment Skillset."""
        tokens = [w.lower() for w in re.findall(r"\w+", raw_content)]

        # Cognitive Skill 1: Extract Entities
        orgs = [o for o in ["State Bank", "NETSOL", "Securities Commission", "Fintech"] if o.lower() in raw_content.lower()]

        # Cognitive Skill 2: Key Phrases
        phrases = [p for p in ["credit policy", "loan default", "interest rate", "collateral requirements"] if p in raw_content.lower()]

        enriched_doc = {
            "id": doc_id,
            "title": title,
            "content": raw_content,
            "tokens": tokens,
            "length": len(tokens),
            "enriched_entities": orgs,
            "enriched_key_phrases": phrases
        }

        self.documents.append(enriched_doc)
        self.doc_lengths[doc_id] = len(tokens)
        self.avg_doc_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        # Build Inverted Index
        for token in set(tokens):
            if token not in self.inverted_index:
                self.inverted_index[token] = []
            self.inverted_index[token].append(doc_id)

    def search_semantic(self, query, top_k=3):
        """
        Executes BM25 / Semantic Scoring search against indexed documents.
        BM25 Formula: IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        """
        query_tokens = [w.lower() for w in re.findall(r"\w+", query)]
        N = len(self.documents)
        k1 = 1.5
        b = 0.75
        scores = {doc["id"]: 0.0 for doc in self.documents}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            matching_doc_ids = self.inverted_index[token]
            df = len(matching_doc_ids)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for doc in self.documents:
                doc_id = doc["id"]
                if doc_id in matching_doc_ids:
                    tf = doc["tokens"].count(token)
                    doc_len = self.doc_lengths[doc_id]
                    denom = tf + k1 * (1.0 - b + b * (doc_len / (self.avg_doc_len + 1e-6)))
                    bm25_token_score = idf * ((tf * (k1 + 1.0)) / denom)
                    scores[doc_id] += bm25_token_score

        # Rank results
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in ranked[:top_k]:
            if score > 0:
                doc_obj = next(d for d in self.documents if d["id"] == doc_id)
                results.append({
                    "id": doc_id,
                    "title": doc_obj["title"],
                    "relevance_score": round(score, 4),
                    "entities": doc_obj["enriched_entities"],
                    "key_phrases": doc_obj["enriched_key_phrases"],
                    "snippet": doc_obj["content"][:110] + "..."
                })
        return results


if __name__ == "__main__":
    print("=" * 65)
    print("Week 12 Task 3: Knowledge Mining with Azure AI Search")
    print("Student: Abubakar Iqbal | NAVTTC Week 12")
    print("=" * 65)

    search_service = AzureCognitiveSearchIndex()

    # Indexing unstructured bank compliance and credit policy documents
    search_service.add_document_with_cognitive_skills(
        doc_id="DOC-01",
        title="2026 Credit Risk & Loan Default Policy",
        raw_content="State Bank regulatory guidelines dictate strict loan default monitoring. Financial institutions must assess borrower checking balances and collateral requirements prior to loan disbursement."
    )
    search_service.add_document_with_cognitive_skills(
        doc_id="DOC-02",
        title="Digital Banking IT Security Architecture",
        raw_content="NETSOL enterprise software integrates OAuth2 authentication, API key rotation, and Azure Key Vault encryption for secure fintech microservices and cloud payment infrastructure."
    )
    search_service.add_document_with_cognitive_skills(
        doc_id="DOC-03",
        title="Retail Mortgage & Interest Rate Schedules",
        raw_content="Annual home financing credit policy and floating interest rate schedules. Borrowers with verifiable employment tenure above four years qualify for discounted margin rates."
    )

    print(f"\n[1] Cognitive Enrichment Pipeline Executed:")
    print(f"Total Documents Cracked & Indexed: {len(search_service.documents)}")
    for d in search_service.documents:
        print(f"\nDocument [{d['id']}]: '{d['title']}'")
        print(f"  Enriched Entities:    {d['enriched_entities']}")
        print(f"  Enriched Key Phrases: {d['enriched_key_phrases']}")

    # 2. Querying Knowledge Mining Index
    queries = [
        "loan default and credit policy",
        "Azure key vault authentication and security"
    ]

    print("\n[2] Executing Azure AI Search Queries (Semantic Inverted Index):")
    for q in queries:
        print(f"\nSearch Query: \"{q}\"")
        matches = search_service.search_semantic(q, top_k=2)
        for rank, m in enumerate(matches, 1):
            print(f"  Rank {rank} (BM25 Score: {m['relevance_score']}): [{m['id']}] {m['title']}")
            print(f"    Snippet: \"{m['snippet']}\"")

    print("\nWeek 12 Task 3 Knowledge Mining Solution Completed.")
