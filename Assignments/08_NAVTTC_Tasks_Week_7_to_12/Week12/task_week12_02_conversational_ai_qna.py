# Week 12 - Task 02: Azure Conversational AI & Question Answering Solution
# NAVTTC Outline Week 12 Days 3 & 4
# Reference: Microsoft Learn / Create Bots with the Bot Framework Composer & Azure AI Language QnA

import re
import difflib

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class AzureConversationalQnAEngine:
    """
    Implements a 2-Tier Azure Conversational AI System:
      1. Conversational Language Understanding (CLU): Classifies user intent and extracts slot entities.
      2. Question Answering (QnA Knowledge Base): Knowledge base of FAQ pairs with fuzzy similarity matching.
    """

    # 1. CLU Intent & Entity Rules
    INTENT_MAP = {
        "CheckBalance": [
            r"check (?:my )?balance", r"what is my account balance", r"how much money do i have", r"account balance"
        ],
        "ApplyLoan": [
            r"apply for (?:a )?loan", r"need (?:a )?personal loan", r"home financing application", r"borrow money"
        ],
        "CardSupport": [
            r"block (?:my )?card", r"lost (?:my )?debit card", r"credit card stolen", r"activate card"
        ],
        "InterestRateInquiry": [
            r"what is the interest rate", r"profit rate on savings", r"mortgage percentage", r"kibor rate"
        ]
    }

    # 2. QnA Knowledge Base
    KNOWLEDGE_BASE = [
        {
            "questions": [
                "What documents are required to apply for a personal loan?",
                "How to apply for credit?",
                "Loan eligibility paperwork"
            ],
            "answer": "To apply for a personal loan, submit your original CNIC, 3-month verified salary slips, 6-month bank statement, and employment proof."
        },
        {
            "questions": [
                "How do I block a compromised or lost debit card?",
                "My card is stolen what should I do?",
                "Freeze my ATM card"
            ],
            "answer": "You can instantly freeze your card via our Mobile Banking App under 'Card Management' or call our 24/7 hotline at 111-222-333 immediately."
        },
        {
            "questions": [
                "What is the current profit rate on savings accounts?",
                "Savings interest rate",
                "How much return on deposit?"
            ],
            "answer": "Our high-yield savings accounts currently offer an annualized profit rate of 12.5% calculated on daily product balance and credited monthly."
        },
        {
            "questions": [
                "What are the branch operating hours?",
                "When does the bank open and close?",
                "Weekend branch timing"
            ],
            "answer": "Branches operate Monday through Friday from 9:00 AM to 5:00 PM. Select commercial hub branches remain open on Saturday from 9:00 AM to 1:30 PM."
        }
    ]

    def predict_intent_and_entities(self, query):
        """CLU Engine: Determines top intent and parses slot values."""
        query_clean = query.lower().strip()
        matched_intent = "None"
        confidence = 0.35

        for intent, patterns in self.INTENT_MAP.items():
            for p in patterns:
                if re.search(p, query_clean):
                    matched_intent = intent
                    confidence = 0.96
                    break
            if matched_intent != "None":
                break

        # Entity Extraction (e.g. monetary amounts or account types)
        entities = []
        amt_match = re.search(r"(\d+(?:,\d+)?(?:\s*(?:k|lac|million|thousand|pkr|rs))?)", query_clean)
        if amt_match:
            entities.append({"category": "Amount", "value": amt_match.group(1)})

        if "savings" in query_clean:
            entities.append({"category": "AccountType", "value": "Savings"})
        elif "current" in query_clean:
            entities.append({"category": "AccountType", "value": "Current"})

        return {
            "topIntent": matched_intent,
            "confidence": confidence,
            "entities": entities
        }

    def query_qna_knowledge_base(self, user_question, threshold=0.45):
        """Azure Question Answering engine using semantic similarity matching."""
        best_match = None
        best_score = 0.0

        for item in self.KNOWLEDGE_BASE:
            for q in item["questions"]:
                ratio = difflib.SequenceMatcher(None, user_question.lower(), q.lower()).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_match = item["answer"]

        if best_score >= threshold:
            return {
                "answer": best_match,
                "confidenceScore": round(best_score, 4),
                "source": "Corporate_Banking_FAQ_KB_v2"
            }
        else:
            return {
                "answer": "I apologize, but I could not find a confident answer in our knowledge base. Would you like me to connect you to a live credit officer?",
                "confidenceScore": round(best_score, 4),
                "source": "Fallback_Handler"
            }


if __name__ == "__main__":
    print("=" * 65)
    print("Week 12 Task 2: Azure Conversational AI & QnA Knowledge Base")
    print("Student: Abubakar Iqbal | NAVTTC Week 12")
    print("=" * 65)

    bot = AzureConversationalQnAEngine()

    # 1. Test Conversational Language Understanding (CLU)
    test_utterances = [
        "I need to apply for a loan of 500k for home renovation",
        "Please check my account balance immediately",
        "My debit card is lost and stolen please freeze it",
        "Hello good afternoon"
    ]

    print("\n[1] Conversational Language Understanding (CLU) Results:")
    for utt in test_utterances:
        clu_res = bot.predict_intent_and_entities(utt)
        print(f"\nUser Utterance: \"{utt}\"")
        print(f"  --> Top Predicted Intent: [{clu_res['topIntent']}] (Confidence: {clu_res['confidence']})")
        print(f"  --> Extracted Entities:   {clu_res['entities']}")

    # 2. Test Question Answering (QnA Knowledge Base)
    faq_queries = [
        "What paperwork do I need for personal credit approval?",
        "How can I block my ATM debit card?",
        "What are the weekend branch timings?",
        "Can I fly an airplane inside the bank?" # Irrelevant query to test fallback
    ]

    print("\n[2] Azure Question Answering (QnA) Engine Matching:")
    for q in faq_queries:
        res = bot.query_qna_knowledge_base(q)
        print(f"\nUser Query: \"{q}\"")
        print(f"  Confidence Match: {res['confidenceScore'] * 100:.1f}% (Source: {res['source']})")
        print(f"  Bot Answer: \"{res['answer']}\"")

    print("\nWeek 12 Task 2 Conversational AI & QnA Solution Completed.")
