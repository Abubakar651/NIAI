# Abubakar Iqbal
# NAVTTC Week 12 - Task 02: Conversational AI & QnA Knowledge Base

import difflib
import re

class ConversationalQnABot:
    INTENTS = {
        "CheckBalance": [r"check balance", r"account balance", r"how much balance"],
        "ApplyLoan": [r"apply for loan", r"need personal loan", r"financing application"],
        "CardSupport": [r"block card", r"lost debit card", r"stolen card"]
    }

    FAQ = [
        {
            "questions": ["What documents are required to apply for a personal loan?", "How to apply for loan", "Loan paperwork"],
            "answer": "Submit original CNIC, 3-month salary slips, and 6-month bank statement."
        },
        {
            "questions": ["How do I block a stolen card?", "Freeze debit card", "Lost card"],
            "answer": "Block your card immediately in the mobile banking app under Card Management or call 111-222-333."
        },
        {
            "questions": ["What is the current savings profit rate?", "Savings interest rate"],
            "answer": "Our high-yield savings account currently offers an annual profit rate of 12.5%."
        }
    ]

    def classify_intent(self, user_msg):
        msg = user_msg.lower()
        for intent, patterns in self.INTENTS.items():
            if any(re.search(p, msg) for p in patterns):
                return intent
        return "GeneralInquiry"

    def answer_faq(self, user_msg):
        best_match = None
        best_ratio = 0.0

        for item in self.FAQ:
            for q in item["questions"]:
                ratio = difflib.SequenceMatcher(None, user_msg.lower(), q.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = item["answer"]

        if best_ratio > 0.40:
            return best_match, round(best_ratio, 2)
        return "I can connect you to an agent for further help.", round(best_ratio, 2)

if __name__ == "__main__":
    print("NAVTTC Week 12 Task 02 - Abubakar Iqbal")

    abubakar_bot = ConversationalQnABot()

    abubakar_queries = [
        "Abubakar wants to check balance in his account",
        "How can Abubakar apply for personal loan paperwork?",
        "My debit card was lost please freeze it"
    ]

    print("\nConversational Intent & QnA Matching:")
    for query in abubakar_queries:
        intent = abubakar_bot.classify_intent(query)
        answer, conf = abubakar_bot.answer_faq(query)
        print(f"\nUser Query: '{query}'")
        print(f" -> Detected Intent: {intent}")
        print(f" -> QnA Answer (Confidence {conf}): {answer}")
