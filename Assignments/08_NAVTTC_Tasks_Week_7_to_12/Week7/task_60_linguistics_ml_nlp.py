# Abubakar Iqbal
# NAVTTC Week 7 - Task 60: Linguistics and NLP Classification

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    print("NAVTTC Week 7 Task 60 - Abubakar Iqbal")

    abubakar_corpus = [
        ("Abubakar loved the fast mobile banking application, money transfer was immediate", "Positive"),
        ("Abubakar received excellent support from customer service for loan processing", "Positive"),
        ("Abubakar appreciates the smooth user interface and intuitive dashboard", "Positive"),
        ("Great profit rates on the savings plan recommended to Abubakar", "Positive"),
        ("The account opening was completed in minutes with zero friction", "Positive"),
        ("Abubakar experienced an application crash while entering his verification pin", "Negative"),
        ("Long delay in customer helpline response, very frustrating experience", "Negative"),
        ("Unrecognized transaction detected on debit card and account locked", "Negative"),
        ("Hidden service charges deducted without clear breakdown", "Negative"),
        ("Slow server response and persistent connection timeout errors", "Negative"),
        ("Abubakar asked what the current interest rate is on vehicle leasing", "Inquiry"),
        ("Where can Abubakar verify the list of required identity documents", "Inquiry"),
        ("What are the working hours for weekend banking branches", "Inquiry"),
        ("How to apply for online international payments activation", "Inquiry"),
        ("Can account statements be downloaded directly in PDF format", "Inquiry")
    ]

    abubakar_df = pd.DataFrame(abubakar_corpus, columns=["text", "category"])
    abubakar_df["cleaned"] = abubakar_df["text"].apply(clean_text)

    abubakar_vec = CountVectorizer()
    X = abubakar_vec.fit_transform(abubakar_df["cleaned"])
    y = abubakar_df["category"]

    abubakar_nb = MultinomialNB()
    abubakar_nb.fit(X, y)

    preds = abubakar_nb.predict(X)
    print(f"Training Accuracy: {accuracy_score(y, preds) * 100:.1f}%")

    test_samples = [
        "Abubakar loves the instant transfer service",
        "The system has continuous error and crash issues",
        "Could you tell me how to apply for loan documents"
    ]
    test_clean = [clean_text(s) for s in test_samples]
    test_preds = abubakar_nb.predict(abubakar_vec.transform(test_clean))

    print("\nPredictions on new samples:")
    for s, p in zip(test_samples, test_preds):
        print(f"'{s}' -> {p}")

if __name__ == "__main__":
    main()
