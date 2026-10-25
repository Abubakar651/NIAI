# NAVTTC Weekly Tasks (Weeks 7 to 12)

**Author:** Abubakar Iqbal  
**Program:** AI, Data Science & Analytics (NAVTTC / NIAI Cohort 2 — NETSOL Technologies)  
**Total Practical Modules:** 30 Tasks Across Weeks 7 to 12  
**Status:** All Tasks Completed and Verified (100% Passed)

---

## Curriculum & Task Mapping

| Week | Directory | Focus & Core Practical Topics | Tasks Covered |
| :--- | :--- | :--- | :---: |
| **Week 7** | [`Week7/`](Week7/) | **Deep Learning I & Boosting**: Neural networks from scratch (2-layer MLP, XOR, backprop), Scikit-Learn `MLPClassifier`, custom Feedforward Neural Network (ReLU, Sigmoid, Softmax), 7 foundational loss functions (MSE, MAE, Huber, BCE, CCE, Hinge, KL), NLP text classification (Linguistics & Bag-of-Words), NLTK text processing (Tokenization, Stemming, Lemmatization, POS tagging), TF-IDF vectorization & Cosine document search, CNN architecture (2D Conv for images and 1D Conv for text sequences), multi-class image feature classification, and Boosting (AdaBoost vs Gradient Boosting). | Tasks 56–64b (10 Tasks) |
| **Week 8** | [`Week8/`](Week8/) | **Deep Learning II (Recurrent Architectures)**: SimpleRNN from scratch with Backpropagation Through Time (BPTT), Long Short-Term Memory (LSTM) cell architecture from scratch (Forget, Input, Candidate, Cell State, Output gates), sequential time-series forecasting with LSTM, Gated Recurrent Unit (GRU) cell from scratch (Reset and Update gates), and empirical benchmark of SimpleRNN vs LSTM vs GRU (parameters and latency). | Week 8 Tasks 1–5 (5 Tasks) |
| **Week 9** | [`Week9/`](Week9/) | **Deep Learning II (Word Embeddings & Sequence Models)**: Word2Vec Continuous Bag-of-Words (CBOW) from scratch, Word2Vec Continuous Skip-Gram from scratch, custom semantic embedding space with vector arithmetic ($v_{\text{King}} - v_{\text{Man}} + v_{\text{Woman}} \approx v_{\text{Queen}}$) and Abubakar's vector nearest neighbors, and the 4 canonical sequence model archetypes (1-to-1, 1-to-Many, Many-to-1, Many-to-Many). | Week 9 Tasks 1–4 (4 Tasks) |
| **Week 10** | [`Week10/`](Week10/) | **Deep Learning II (Advanced Sequence Models & Capstone Architecture)**: Bi-Directional LSTM from scratch (forward and backward hidden state fusion), Scaled Dot-Product Attention mechanism ($Q, K, V$, softmax attention weights, context vector), and End-to-End Enterprise Capstone Architecture blueprint for an intelligent Credit Risk decision engine. | Week 10 Tasks 1–3 (3 Tasks) |
| **Week 11** | [`Week11/`](Week11/) | **Microsoft Azure AI Services (Task 65 / Cognitive Services & Security)**: Azure AI Service Selector (Vision, Speech, Language, Decision, Applied AI), Azure Security and Governance (API Key rotation, Entra ID Managed Identity, VNet firewall, Responsible AI 6 pillars), Azure AI Vision & Read OCR solution (document parsing, invoice/salary slip extraction), and Azure Decision Cognitive Services (Time-series Anomaly Detector & Content Moderator PII/profanity screening). | Week 11 Tasks 1–4 (4 Tasks) |
| **Week 12** | [`Week12/`](Week12/) | **Microsoft Azure AI Services (NLP, Conversational AI & CI/CD Deployment)**: Azure AI Language (Sentiment analysis, Key Phrase extraction, Named Entity Recognition NER, translation), Azure Conversational Language Understanding (CLU) & QnA Maker Knowledge Base, Knowledge Mining with Azure AI Search (BM25 scoring and inverted indexing), and Production Deployment (Docker Compose containerization, GitHub Actions CI/CD workflow, and FinOps cost telemetry monitoring). | Week 12 Tasks 1–4 (4 Tasks) |

---

## How to Run Tasks

### 1. Execute Any Single Task
Activate your virtual environment and run the script:
```bash
# Example 1: Run Neural Network from Scratch
python Week7/task_56_neural_network_scratch.py

# Example 2: Run LSTM Sequence Predictor
python Week8/task_week8_03_lstm_sequence_prediction.py

# Example 3: Run Word2Vec CBOW
python Week9/task_week9_01_word2vec_cbow.py

# Example 4: Run Scaled Dot-Product Attention
python Week10/task_week10_02_attention_mechanism.py

# Example 5: Run Azure AI Vision & OCR
python Week11/task_week11_03_azure_computer_vision_ocr.py

# Example 6: Run Azure Knowledge Mining with Cognitive Search
python Week12/task_week12_03_azure_cognitive_search.py
```

### 2. Run the Full Automated Test Suite
To execute all 30 practical scripts in sequence and verify their outputs:
```bash
python run_all_tasks.py
```
