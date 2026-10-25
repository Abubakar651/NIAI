# Abubakar Iqbal
# NAVTTC Week 11 - Task 01: Microsoft Azure AI Service Selection

import pandas as pd

class AzureAIServiceSelector:
    SERVICES = {
        "Vision": {
            "Azure AI Vision": "Image analysis, object detection, OCR",
            "Custom Vision": "Custom image classification and object detection",
            "Face API": "Facial recognition and liveness detection"
        },
        "Language": {
            "Azure AI Language": "Sentiment, NER, key phrase extraction, PII masking",
            "CLU": "Conversational Language Understanding for bots",
            "Azure AI Translator": "Multilingual translation across 100+ languages"
        },
        "Speech": {
            "Speech to Text": "Transcribing audio and phone conversations",
            "Text to Speech": "Synthesizing natural neural voice",
            "Speech Translation": "Real-time spoken translation"
        },
        "Decision": {
            "Anomaly Detector": "Monitoring time-series metrics and fraud velocity",
            "Content Moderator": "Screening text and images for policy compliance",
            "Personalizer": "Reinforcement learning for recommendations"
        },
        "Applied AI": {
            "Document Intelligence": "Parsing invoices, receipts, and ID cards",
            "Azure AI Search": "Enterprise search with cognitive enrichment"
        }
    }

    def select(self, modality):
        for mod, svcs in self.SERVICES.items():
            if mod.lower() == modality.strip().lower():
                return svcs
        return {"Azure AI Multi-Service": "All cognitive services unified"}

if __name__ == "__main__":
    print("NAVTTC Week 11 Task 01 - Abubakar Iqbal")

    abubakar_selector = AzureAIServiceSelector()

    scenarios = [
        ("Vision", "Salary slip parsing and national identity verification"),
        ("Language", "Analyzing customer reviews and feedback sentiment"),
        ("Decision", "Detecting fraudulent ATM transaction spikes"),
        ("Applied AI", "Building searchable archive of financial policies")
    ]

    records = []
    for mod, desc in scenarios:
        options = abubakar_selector.select(mod)
        records.append({
            "Requirement": desc,
            "Modality": mod,
            "Primary Service": list(options.keys())[0],
            "Details": list(options.values())[0]
        })

    abubakar_df = pd.DataFrame(records)
    print("\nAzure AI Service Selection Matrix:")
    print(abubakar_df.to_string(index=False))
