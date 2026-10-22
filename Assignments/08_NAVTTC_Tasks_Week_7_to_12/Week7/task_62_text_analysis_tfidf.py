# Task 62 - Text Analysis (TF-IDF Vectorization & Sentiment Analysis)
# NAVTTC Outline Week 7 / Task 62
# Reference: https://medium.com/towards-artificial-intelligence/natural-language-processing-nlp-withpython-tutorial-for-beginners-1f54e610a1a0

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

def run_text_analysis():
    print("=" * 65)
    print("Task 62: Advanced Text Analysis (TF-IDF, N-Grams & Cosine Search)")
    print("Student: Abubakar Iqbal | NAVTTC Week 7")
    print("=" * 65)

    # Document Corpus for Information Retrieval & TF-IDF
    corpus = [
        "Artificial intelligence and deep learning drive automated credit decision systems.",
        "Deep learning neural networks require large datasets and high computational power.",
        "Credit risk models evaluate borrower income, credit history, and default probability.",
        "Automated loan approval algorithms reduce banking operational costs and turnaround time.",
        "Natural language processing enables intelligent chatbots to handle financial customer inquiries.",
        "Fraud detection architectures analyze real-time debit card transaction streams."
    ]

    # 1. TF-IDF Representation with Unigrams and Bigrams
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    print(f"\n[1] TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"Total Unique Unigram + Bigram Features: {len(feature_names)}")
    print("Sample Features:", feature_names[:10])

    # 2. Semantic Search using Cosine Similarity
    user_query = "credit default and risk evaluation models"
    query_vec = tfidf_vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    print(f"\n[2] Semantic Document Search for Query: \"{user_query}\"")
    ranking = np.argsort(similarities)[::-1]
    for rank, doc_idx in enumerate(ranking[:3], 1):
        print(f"  Rank {rank} (Cosine Similarity: {similarities[doc_idx]:.4f}):")
        print(f"    Document {doc_idx + 1}: \"{corpus[doc_idx]}\"")

    # 3. Sentiment Analysis on Financial / Service Reviews
    sentiment_data = [
        ("The banking app is fast, intuitive, and extremely helpful.", 1),
        ("Excellent loan approval speed and professional customer service.", 1),
        ("Great rates on car financing and seamless transaction experience.", 1),
        ("I love the new mobile check deposit feature, worked perfectly.", 1),
        ("Customer support is polite and resolved my query in five minutes.", 1),
        ("The app keeps crashing and logged me out during payment.", 0),
        ("Terrible service, high hidden fees and unresponsive phone helpline.", 0),
        ("Worst credit card experience, unauthorized charges were not refunded.", 0),
        ("Very slow loading times, account balance took hours to update.", 0),
        ("Disappointing service, my account was locked for three days without reason.", 0),
        ("Average service, mobile app is okay but website is outdated.", 1),
        ("Frequent system maintenance during working hours is very annoying.", 0)
    ]

    df_reviews = pd.DataFrame(sentiment_data, columns=["review", "sentiment"])
    
    # TF-IDF on review texts
    sentiment_vec = TfidfVectorizer(stop_words="english", max_features=50)
    X_reviews = sentiment_vec.fit_transform(df_reviews["review"])
    y_reviews = df_reviews["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X_reviews, y_reviews, test_size=0.33, random_state=42)

    # Train Logistic Regression sentiment classifier
    sentiment_clf = LogisticRegression(random_state=42)
    sentiment_clf.fit(X_train, y_train)

    y_pred = sentiment_clf.predict(X_test)
    print(f"\n[3] Sentiment Classification Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
    print("\nClassification Report (Sentiment Model):")
    print(classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"], zero_division=0))

    # Real-world test on unseen review
    new_reviews = [
        "Amazing mobile banking experience, totally love it!",
        "Poor customer care and terrible hidden transaction costs."
    ]
    new_vecs = sentiment_vec.transform(new_reviews)
    preds = sentiment_clf.predict(new_vecs)
    print("\n[4] Sentiment Inference on New Unseen Feedback:")
    for rev, p in zip(new_reviews, preds):
        label = "Positive (1)" if p == 1 else "Negative (0)"
        print(f"  \"{rev}\" --> Predicted: [{label}]")

if __name__ == "__main__":
    run_text_analysis()
