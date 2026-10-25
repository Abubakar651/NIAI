# Abubakar Iqbal
# NAVTTC Week 7 - Task 61: Text Processing Primitives

import re
import string

class TextProcessor:
    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "just", "me", "more", "most", "my", "no", "nor",
        "not", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "same",
        "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
        "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
        "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "with", "you", "your"
    }

    LEMMAS = {
        "running": "run", "runs": "run", "ran": "run",
        "studying": "study", "studies": "study", "studied": "study",
        "learning": "learn", "learns": "learn", "learned": "learn",
        "investing": "invest", "invests": "invest", "invested": "invest",
        "banking": "bank", "banks": "bank", "banked": "bank",
        "borrowed": "borrow", "borrowing": "borrow", "loans": "loan",
        "transactions": "transaction", "transacted": "transact"
    }

    @staticmethod
    def sent_tokenize(text):
        return [s.strip() for s in re.split(r"(?<=[.!?]) +", text) if s.strip()]

    @staticmethod
    def word_tokenize(text):
        cleaned = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        return [w.strip() for w in cleaned.split() if w.strip()]

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t.lower() not in self.STOPWORDS]

    @staticmethod
    def stem(word):
        w = word.lower()
        if w.endswith("sses"):
            return w[:-2]
        if w.endswith("ies"):
            return w[:-2]
        if w.endswith("ss"):
            return w
        if w.endswith("s") and len(w) > 2:
            return w[:-1]
        if w.endswith("ing") and len(w) > 4:
            return w[:-3]
        if w.endswith("ed") and len(w) > 3:
            return w[:-2]
        return w

    def lemmatize(self, word):
        w = word.lower()
        return self.LEMMAS.get(w, w)

    @staticmethod
    def pos_tag(tokens):
        tags = []
        verbs = {"study", "studying", "learn", "learning", "build", "building", "train", "training", "apply", "applying"}
        for word in tokens:
            w = word.lower()
            if w in verbs:
                tag = "VERB"
            elif w in {"abubakar", "iqbal", "lahore", "python", "navttc"}:
                tag = "PROPN"
            elif w in {"deep", "smart", "fast", "accurate"}:
                tag = "ADJ"
            else:
                tag = "NOUN"
            tags.append((word, tag))
        return tags

if __name__ == "__main__":
    print("NAVTTC Week 7 Task 61 - Abubakar Iqbal")

    processor = TextProcessor()
    abubakar_text = (
        "Abubakar Iqbal is studying artificial intelligence and deep learning at NAVTTC in Lahore. "
        "Abubakar is building neural models and learning advanced text processing techniques. "
        "His algorithms handle complex financial transactions and help students learn faster!"
    )

    abubakar_sentences = processor.sent_tokenize(abubakar_text)
    print(f"\nSentences ({len(abubakar_sentences)}):")
    for idx, s in enumerate(abubakar_sentences, 1):
        print(f" {idx}. {s}")

    abubakar_words = processor.word_tokenize(abubakar_text)
    print(f"\nTotal Words: {len(abubakar_words)}")

    abubakar_filtered = processor.remove_stopwords(abubakar_words)
    print(f"Words after Stopword Removal: {len(abubakar_filtered)}")

    print("\nStemming vs Lemmatization:")
    sample_words = ["studying", "learning", "transactions", "invested", "borrowing"]
    for w in sample_words:
        print(f"{w:<14} -> Stem: {processor.stem(w):<12} | Lemma: {processor.lemmatize(w)}")

    print("\nPOS Tagging Sample:")
    for word, tag in processor.pos_tag(abubakar_filtered[:8]):
        print(f"{word:<12} : {tag}")
