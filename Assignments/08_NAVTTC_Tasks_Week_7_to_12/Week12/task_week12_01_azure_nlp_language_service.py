# Abubakar Iqbal
# NAVTTC Week 12 - Task 01: Azure AI Language Service (NLP & Translation)

import re

class AzureLanguageClient:
    POSITIVE_WORDS = {"excellent", "love", "fast", "reliable", "secure", "great", "helpful"}
    NEGATIVE_WORDS = {"slow", "crash", "horrible", "delay", "error", "unresponsive"}

    def analyze_sentiment(self, text):
        words = set(re.findall(r"\w+", text.lower()))
        pos = len(words.intersection(self.POSITIVE_WORDS))
        neg = len(words.intersection(self.NEGATIVE_WORDS))
        total = pos + neg + 1e-6
        pos_score = round(pos / total, 2)
        neg_score = round(neg / total, 2)

        sentiment = "positive" if pos >= neg else "negative"
        return {"sentiment": sentiment, "positive": pos_score, "negative": neg_score}

    @staticmethod
    def extract_entities(text):
        entities = []
        if re.search(r"\bAbubakar Iqbal\b", text, flags=re.IGNORECASE):
            entities.append({"text": "Abubakar Iqbal", "category": "Person"})
        if re.search(r"\bNETSOL Technologies\b", text, flags=re.IGNORECASE):
            entities.append({"text": "NETSOL Technologies", "category": "Organization"})
        if re.search(r"\bLahore\b", text, flags=re.IGNORECASE):
            entities.append({"text": "Lahore", "category": "Location"})
        for amt in re.findall(r"PKR\s*[\d,]+", text):
            entities.append({"text": amt, "category": "Currency"})
        return entities

    @staticmethod
    def translate(text, target="ur"):
        translations = {
            "loan approved": "قرض منظور کر دیا گیا ہے",
            "account balance": "اکاؤنٹ کا بیلنس",
            "payment successful": "ادائیگی کامیاب رہی"
        }
        return translations.get(text.lower().strip(), f"[{target.upper()}]: {text}")

if __name__ == "__main__":
    print("NAVTTC Week 12 Task 01 - Abubakar Iqbal")

    abubakar_nlp = AzureLanguageClient()

    abubakar_message = (
        "Abubakar Iqbal submitted his commercial financing application to NETSOL Technologies in Lahore. "
        "The requested amount of PKR 1,500,000 was approved with excellent speed and reliable support! "
        "I love this fast banking platform."
    )

    print(f"\nDocument:\n\"{abubakar_message}\"")

    sent = abubakar_nlp.analyze_sentiment(abubakar_message)
    print(f"\nSentiment: {sent['sentiment'].upper()} (Pos: {sent['positive']}, Neg: {sent['negative']})")

    entities = abubakar_nlp.extract_entities(abubakar_message)
    print("\nNamed Entities Extracted:")
    for e in entities:
        print(f" - {e['text']:<22} [{e['category']}]")

    print("\nAzure AI Translation (EN -> UR):")
    sample_text = "Loan Approved"
    print(f"'{sample_text}' -> '{abubakar_nlp.translate(sample_text, 'ur')}'")
