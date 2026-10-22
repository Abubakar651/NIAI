# Week 11 - Task 01: Microsoft Azure AI Service Selection & Decision Engine
# NAVTTC Outline Week 11 Day 1 & Task 65
# Reference: Microsoft Learn / AI-102: Designing and Implementing an Azure AI Solution

import json
import pandas as pd

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class AzureAIServiceSelector:
    """
    Expert decision tree system for selecting appropriate Microsoft Azure AI Services
    based on business domain, input modality, and compliance requirements.
    """

    CATALOG = {
        "Vision": {
            "Azure AI Vision": "General image analysis, object detection, thumbnail generation, OCR text extraction.",
            "Custom Vision": "Domain-specific customized image classification and targeted defect/brand detection.",
            "Face API": "Facial verification, liveness detection, and identity authentication (strictly controlled by Responsible AI)."
        },
        "Language": {
            "Azure AI Language": "Sentiment analysis, entity recognition (NER), key phrase extraction, and PII masking.",
            "Conversational Language Understanding (CLU)": "Intent determination and entity extraction for chatbots.",
            "Azure AI Translator": "Real-time multilingual translation across 100+ languages."
        },
        "Speech": {
            "Speech to Text (STT)": "Transcribing customer support phone calls with custom acoustic/language models.",
            "Text to Speech (TTS)": "Synthesizing human-like neural audio voices for interactive voice response (IVR).",
            "Speech Translation": "Simultaneous spoken language translation."
        },
        "Decision": {
            "Anomaly Detector": "Ingesting time-series transactions to flag unusual fraudulent spikes.",
            "Content Moderator": "Automated text, image, and video screening for offensive content and PII leakage.",
            "Personalizer": "Reinforcement learning recommendation engine optimizing customer offer conversions."
        },
        "Applied AI": {
            "Azure AI Document Intelligence (Form Recognizer)": "Prebuilt receipt, invoice, and ID card key-value parsing.",
            "Azure AI Search": "Enterprise knowledge mining with cognitive enrichment pipelines."
        }
    }

    def recommend_service(self, modality, requirements):
        """Recommends specific Azure service matching modality and technical constraints."""
        target_modality = None
        for k in self.CATALOG.keys():
            if k.lower() == modality.strip().lower():
                target_modality = k
                break

        if not target_modality:
            return {
                "Recommended Service": "Azure AI Services (Multi-service Resource)",
                "Modality": modality,
                "Service Capability": "General multi-modal cognitive resource.",
                "Deployment Option": "Azure Cloud"
            }

        services = self.CATALOG[target_modality]
        for name, desc in services.items():
            if any(req.lower() in desc.lower() for req in requirements):
                return {
                    "Recommended Service": name,
                    "Modality": target_modality,
                    "Service Capability": desc,
                    "Deployment Option": "Azure Cloud Endpoint or Disconnected Docker Container"
                }

        first_svc = list(services.keys())[0]
        return {
            "Recommended Service": first_svc,
            "Modality": target_modality,
            "Service Capability": services[first_svc],
            "Deployment Option": "Azure Cloud Endpoint or Disconnected Docker Container"
        }


if __name__ == "__main__":
    print("=" * 65)
    print("Week 11 Task 1: Microsoft Azure AI Service Selection Framework")
    print("Student: Abubakar Iqbal | NAVTTC Week 11 / Task 65")
    print("=" * 65)

    selector = AzureAIServiceSelector()

    # Enterprise Banking Use Cases
    use_cases = [
        ("Vision", ["ocr", "text extraction", "receipt"], "Extract text from scanned loan applicant salary slips and ID cards"),
        ("Decision", ["fraud", "anomaly", "time-series"], "Flag sudden anomalous transaction volume on debit cards"),
        ("Language", ["sentiment", "pii", "entity"], "Detect customer dissatisfaction and mask CNIC/SSN numbers in banking chat"),
        ("Speech", ["transcribing", "phone", "calls"], "Convert call center audio recordings into searchable text transcripts"),
        ("Applied AI", ["invoice", "receipt"], "Automate invoice accounting and table extraction")
    ]

    print("\n--- Mapping Business Scenarios to Microsoft Azure AI Services ---")
    recommendations = []
    for modality, reqs, scenario in use_cases:
        rec = selector.recommend_service(modality, reqs)
        recommendations.append({
            "Business Scenario": scenario,
            "Modality": rec["Modality"],
            "Recommended Azure Service": rec["Recommended Service"]
        })

    df_recs = pd.DataFrame(recommendations)
    print(df_recs.to_string(index=False))

    print("\nFull Azure AI Catalog Structure:")
    for cat, svcs in selector.CATALOG.items():
        print(f"\n[{cat}]")
        for s_name in svcs.keys():
            print(f"  - {s_name}")

    print("\nWeek 11 Task 1 Azure AI Service Selection Completed.")
