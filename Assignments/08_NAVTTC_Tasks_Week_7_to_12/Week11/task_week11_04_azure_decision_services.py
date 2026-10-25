# Abubakar Iqbal
# NAVTTC Week 11 - Task 04: Azure Decision Services (Anomaly Detector & Content Moderator)

import pandas as pd
import re

class DecisionServices:
    @staticmethod
    def detect_anomalies(series, k=2.0):
        s = pd.Series(series)
        mean = s.rolling(window=5, min_periods=1).mean()
        std = s.rolling(window=5, min_periods=1).std().fillna(0.1)
        upper = mean + (k * std)
        lower = mean - (k * std)

        records = []
        for i in range(len(series)):
            val = series[i]
            is_anom = val > upper[i] or val < lower[i]
            records.append({
                "index": i + 1,
                "value": val,
                "expected": round(mean[i], 1),
                "upper": round(upper[i], 1),
                "anomaly": is_anom
            })
        return records

    @staticmethod
    def moderate_text(text):
        profanity = {"scam", "cheat", "fraudster", "idiot", "terrible"}
        clean = text
        flagged_words = []
        for word in profanity:
            if re.search(rf"\b{word}\b", clean, flags=re.IGNORECASE):
                flagged_words.append(word)
                clean = re.sub(rf"\b{word}\b", "***", clean, flags=re.IGNORECASE)

        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", clean)
        phones = re.findall(r"\b03\d{2}[-\s]?\d{7}\b", clean)

        clean = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", clean)
        clean = re.sub(r"\b03\d{2}[-\s]?\d{7}\b", "[REDACTED_PHONE]", clean)

        return {
            "censored": clean,
            "flagged_terms": flagged_words,
            "detected_pii": {"emails": emails, "phones": phones},
            "requires_review": len(flagged_words) > 0 or len(phones) > 0
        }

if __name__ == "__main__":
    print("NAVTTC Week 11 Task 04 - Abubakar Iqbal")

    abubakar_tx_stream = [110, 115, 120, 118, 125, 122, 850, 120, 115, 112]
    anomalies = DecisionServices.detect_anomalies(abubakar_tx_stream, k=2.0)

    print("\nAnomaly Detection on Transaction Velocity:")
    for a in anomalies:
        flag = " [!] SPIKE DETECTED" if a["anomaly"] else ""
        print(f"Step {a['index']:2d}: Value = {a['value']:3d} | Expected = {a['expected']:5.1f}{flag}")

    abubakar_feedback = (
        "Customer service was terrible and I felt this is a scam! "
        "Please contact Abubakar Iqbal at 0300-1234567 or email at abubakar.iqbal@domain.com."
    )

    print("\nContent Moderation & PII Redaction:")
    mod_res = DecisionServices.moderate_text(abubakar_feedback)
    print(f"Censored: {mod_res['censored']}")
    print(f"Flagged Words: {mod_res['flagged_terms']}")
    print(f"Extracted PII: {mod_res['detected_pii']}")
    print(f"Review Required: {mod_res['requires_review']}")
