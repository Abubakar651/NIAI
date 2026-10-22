# Task 60 - Linguistics using Machine Learning in Python
# NAVTTC Outline Week 7 / Task 60
# Reference: https://medium.com/towards-artificial-intelligence/natural-language-processing-nlp-withpython-tutorial-for-beginners-1f54e610a1a0

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class LinguisticsMLClassifier:
    """
    End-to-End Linguistics & NLP Classification System.
    Implements: Text normalization, Regex Token Cleaning, Bag-of-Words Vectorization,
    and Probabilistic Linguistic Modeling via Multinomial Naive Bayes.
    """

    @staticmethod
    def clean_text(text):
        # Remove punctuation, digits, and convert to lower case
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def run(self):
        print("=" * 65)
        print("Task 60: Linguistics and Text Classification using NLP & ML")
        print("Student: Abubakar Iqbal | NAVTTC Week 7")
        print("=" * 65)

        # Corpus: Customer sentiment & Intent across Retail Banking / Fintech
        raw_corpus = [
            ("I love this banking mobile app, transactions are instant and smooth!", "Positive"),
            ("Outstanding customer support, they resolved my loan issue quickly.", "Positive"),
            ("Great user interface and wonderful investment features.", "Positive"),
            ("Highly recommend this savings account, highest profit rate in market.", "Positive"),
            ("Seamless experience with credit card payments, five stars!", "Positive"),
            ("The application keeps crashing every time I enter OTP, terrible experience.", "Negative"),
            ("Horrible customer care, waiting on hold for two hours with no response.", "Negative"),
            ("Fraudulent transaction detected and account got frozen without notice.", "Negative"),
            ("Worst exchange rates and hidden service fees on every transfer.", "Negative"),
            ("Very slow server, login timeout error continuously happening.", "Negative"),
            ("Could you please tell me the interest rate on home financing?", "Inquiry"),
            ("What is the branch timing on Saturday and public holidays?", "Inquiry"),
            ("How do I request a new debit card checkbook online?", "Inquiry"),
            ("Where can I download the annual account statement PDF?", "Inquiry"),
            ("Can you verify if foreign currency remittances are accepted?", "Inquiry")
        ]

        df = pd.DataFrame(raw_corpus, columns=["text", "category"])
        df["cleaned_text"] = df["text"].apply(self.clean_text)

        print("\n--- Raw Sample Sentences & Linguistic Cleaning ---")
        for i in range(3):
            print(f"Original: {df['text'].iloc[i]}")
            print(f"Cleaned:  {df['cleaned_text'].iloc[i]}\n")

        # Vectorization: Bag of Words representation
        vectorizer = CountVectorizer(stop_words="english", max_features=100)
        X = vectorizer.fit_transform(df["cleaned_text"]).toarray()
        y = df["category"]

        print(f"Vocabulary Size: {len(vectorizer.get_feature_names_out())} Unique Linguistic Tokens")
        print("Sample Vocabulary Tokens:", vectorizer.get_feature_names_out()[:12])

        # Train / Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

        # Multinomial Naive Bayes (Standard linguistic text classifier)
        classifier = MultinomialNB(alpha=1.0) # Laplace smoothing
        classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\nModel Performance on Unseen Linguistic Test Sentences:")
        print(f"Classification Accuracy: {acc * 100:.1f}%")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nDetailed Linguistic Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Real-time inference on new linguistic prompts
        test_samples = [
            "Fantastic customer care and very fast deposit processing!",
            "Application crashed and deducted money without confirming payment.",
            "What is the procedure to open a foreign currency account?"
        ]
        test_cleaned = [self.clean_text(s) for s in test_samples]
        test_vectors = vectorizer.transform(test_cleaned).toarray()
        inferred = classifier.predict(test_vectors)

        print("\n--- Real-Time Linguistic Inference on New User Inputs ---")
        for text, pred in zip(test_samples, inferred):
            print(f"User Input: \"{text}\" --> Classified Category: [{pred}]")

if __name__ == "__main__":
    clf = LinguisticsMLClassifier()
    clf.run()
