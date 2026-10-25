# Abubakar Iqbal
# NAVTTC Week 7 - Task 62: TF-IDF Text Analysis and Similarity

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    print("NAVTTC Week 7 Task 62 - Abubakar Iqbal")

    abubakar_corpus = [
        "Abubakar Iqbal designs deep learning neural networks for sequence modeling and NLP.",
        "Deep learning architectures require matrix computations and labeled training datasets.",
        "Abubakar builds automated credit scoring engines evaluating risk and borrower income.",
        "Loan approval systems reduce manual evaluation time and improve operational accuracy.",
        "Natural language processing allows conversational chatbots to address customer banking questions.",
        "Real-time fraud detection analyzes transactional data streams to flag suspicious activity."
    ]

    abubakar_tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    abubakar_matrix = abubakar_tfidf.fit_transform(abubakar_corpus)
    vocab = abubakar_tfidf.get_feature_names_out()

    print(f"\nTF-IDF Matrix Shape: {abubakar_matrix.shape}")
    print(f"Sample Features: {list(vocab[:8])}")

    abubakar_query = "Abubakar neural network sequence models"
    query_vec = abubakar_tfidf.transform([abubakar_query])
    similarities = cosine_similarity(query_vec, abubakar_matrix).flatten()

    print(f"\nCosine Search Query: '{abubakar_query}'")
    ranking = np.argsort(similarities)[::-1]
    for rank, idx in enumerate(ranking[:3], 1):
        print(f"Rank {rank} (Score: {similarities[idx]:.4f}) -> {abubakar_corpus[idx]}")

    abubakar_reviews = [
        ("The new interface is fast and easy to navigate", 1),
        ("Excellent loan disbursement speed and helpful customer support", 1),
        ("Great mobile application features for quick account transfers", 1),
        ("I enjoy using the digital banking platform everyday", 1),
        ("The login session crashes repeatedly during payment verification", 0),
        ("Terrible experience with excessive hidden deduction fees", 0),
        ("Customer service line remained busy with zero assistance provided", 0),
        ("The transfer failed and deducted balance without notification", 0)
    ]

    df_rev = pd.DataFrame(abubakar_reviews, columns=["text", "label"])
    vec_clf = TfidfVectorizer(stop_words="english")
    X_train = vec_clf.fit_transform(df_rev["text"])
    y_train = df_rev["label"]

    abubakar_clf = LogisticRegression(random_state=42)
    abubakar_clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, abubakar_clf.predict(X_train))
    print(f"\nSentiment Classifier Accuracy: {train_acc * 100:.1f}%")

    test_doc = ["Abubakar found the payment system quick and helpful"]
    pred = abubakar_clf.predict(vec_clf.transform(test_doc))[0]
    sentiment_str = "Positive" if pred == 1 else "Negative"
    print(f"Test Review: '{test_doc[0]}' -> Sentiment: {sentiment_str}")

if __name__ == "__main__":
    main()
