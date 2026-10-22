# Week 11 - Task 04: Azure Cognitive Decision Services (Anomaly Detection & Content Moderation)
# NAVTTC Outline Week 11 Day 4 & 5
# Reference: Microsoft Learn / Azure Anomaly Detector & Azure Content Moderator

import numpy as np
import pandas as pd
import re

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class AzureDecisionServicesSuite:
    """
    Implements two core Azure Decision Cognitive Services:
      1. Azure Anomaly Detector: Time-series statistical deviation modeling to detect anomalous credit card velocity.
      2. Azure Content Moderator: Screening user reviews and chat inputs for profanity, hate speech, and sensitive PII.
    """

    # --- 1. Azure Anomaly Detector Engine ---
    @staticmethod
    def detect_time_series_anomalies(series, sensitivity=95):
        """
        Simulates Azure Anomaly Detector 'entire/detect' endpoint using
        rolling statistical margins (Mean +/- k * StdDev).
        sensitivity: 0-99 (higher sensitivity flags smaller deviations)
        """
        k = 3.0 - (sensitivity / 100.0) * 1.5  # maps 95 sensitivity to ~1.57 std dev
        rolling_mean = pd.Series(series).rolling(window=5, min_periods=1).mean()
        rolling_std = pd.Series(series).rolling(window=5, min_periods=1).std().fillna(0.1)

        upper_margin = rolling_mean + (k * rolling_std)
        lower_margin = rolling_mean - (k * rolling_std)

        is_anomaly = (series > upper_margin) | (series < lower_margin)

        results = []
        for i in range(len(series)):
            results.append({
                "timestamp_idx": i + 1,
                "value": round(series[i], 2),
                "expected_value": round(rolling_mean[i], 2),
                "upper_margin": round(upper_margin[i], 2),
                "is_anomaly": bool(is_anomaly[i]),
                "severity": round(abs(series[i] - rolling_mean[i]) / (rolling_std[i] + 1e-6), 2) if is_anomaly[i] else 0.0
            })
        return results

    # --- 2. Azure Content Moderator Engine ---
    @staticmethod
    def moderate_text_content(text):
        """
        Simulates Azure Content Moderator 'ProcessText' endpoint:
        Detects PII (Emails, Phones, Credit Cards) and inappropriate language.
        """
        # PII Detection Patterns
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        phone_pattern = r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b|\b03\d{2}[-\s]?\d{7}\b"
        card_pattern = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"

        # Restricted Profanity Term Lexicon
        profanity_lexicon = {"scam", "fraudster", "cheat", "bastard", "idiot", "hell", "damn"}

        emails_found = re.findall(email_pattern, text)
        phones_found = re.findall(phone_pattern, text)
        cards_found = re.findall(card_pattern, text)

        words = set(re.findall(r"\w+", text.lower()))
        offensive_terms = list(words.intersection(profanity_lexicon))

        # Redact PII and offensive words
        clean_text = text
        for email in emails_found:
            clean_text = clean_text.replace(email, "[PII-EMAIL-REDACTED]")
        for phone in phones_found:
            clean_text = clean_text.replace(phone, "[PII-PHONE-REDACTED]")
        for card in cards_found:
            clean_text = clean_text.replace(card, "[PII-CARD-REDACTED]")
        for term in offensive_terms:
            clean_text = re.sub(re.escape(term), "***", clean_text, flags=re.IGNORECASE)

        is_flagged = len(offensive_terms) > 0 or len(cards_found) > 0

        return {
            "original_text": text,
            "censored_text": clean_text,
            "is_flagged_for_human_review": is_flagged,
            "detected_pii": {
                "emails": emails_found,
                "phone_numbers": phones_found,
                "credit_cards": cards_found
            },
            "offensive_terms_detected": offensive_terms
        }


if __name__ == "__main__":
    print("=" * 65)
    print("Week 11 Task 4: Azure Decision Services (Anomaly & Content Moderator)")
    print("Student: Abubakar Iqbal | NAVTTC Week 11")
    print("=" * 65)

    suite = AzureDecisionServicesSuite()

    # 1. Anomaly Detector on Financial Transaction Velocity
    # Baseline normal transactions: 100-140/hour, with sudden spike to 850 (flash fraud attempt)
    transactions = [105, 112, 118, 122, 115, 120, 128, 850, 125, 119, 110, 114, 15, 118]
    print("\n[1] Azure Anomaly Detector: Real-Time Transaction Stream Inspection:")
    anomalies = suite.detect_time_series_anomalies(transactions, sensitivity=95)

    df_anom = pd.DataFrame(anomalies)
    flagged = df_anom[df_anom["is_anomaly"] == True]
    print(f"Total Stream Intervals Analyzed: {len(transactions)}")
    print(f"Anomalies Detected: {len(flagged)}")
    for _, row in flagged.iterrows():
        print(f"  --> ALERT at Interval {int(row['timestamp_idx'])}: Observed Value = {row['value']} (Expected ~{row['expected_value']}, Upper = {row['upper_margin']}) | Severity: {row['severity']}")

    # 2. Azure Content Moderator on User Feedback
    user_comment = (
        "This bank is a total scam run by a cheat! Call me at 0300-1234567 or email at "
        "victim@testmail.com. You charged my card 4532-8901-2345-6789 without my consent!"
    )

    print("\n[2] Azure Content Moderator: Text & PII Screening:")
    mod_result = suite.moderate_text_content(user_comment)
    print("Original User Input:\n ", mod_result["original_text"])
    print("\nCensored & Sanitized Output:\n ", mod_result["censored_text"])
    print(f"\nFlagged for Compliance Review: {mod_result['is_flagged_for_human_review']}")
    print(f"Detected PII Elements: {mod_result['detected_pii']}")
    print(f"Offensive Language Detected: {mod_result['offensive_terms_detected']}")

    print("\nWeek 11 Task 4 Azure Decision Services Pipeline Complete.")
