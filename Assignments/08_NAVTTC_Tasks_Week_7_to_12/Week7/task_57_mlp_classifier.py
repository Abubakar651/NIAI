# Abubakar Iqbal
# NAVTTC Week 7 - Task 57: MLP Classifier

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("NAVTTC Week 7 Task 57 - Abubakar Iqbal")

    np.random.seed(42)
    n = 300

    income = np.random.uniform(25000, 120000, n)
    credit_score = np.random.uniform(300, 850, n)
    dti = np.random.uniform(0.1, 0.6, n)
    savings = np.random.uniform(1000, 50000, n)
    age = np.random.uniform(20, 60, n)

    score = (credit_score / 850) * 0.4 + (savings / 50000) * 0.3 - dti * 0.3
    labels = np.zeros(n, dtype=int)
    labels[score > 0.45] = 0
    labels[(score <= 0.45) & (score >= 0.25)] = 1
    labels[score < 0.25] = 2

    abubakar_df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "debt_to_income": dti,
        "savings": savings,
        "age": age,
        "risk_tier": labels
    })

    X = abubakar_df.drop(columns=["risk_tier"])
    y = abubakar_df["risk_tier"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    abubakar_scaler = StandardScaler()
    X_train_scaled = abubakar_scaler.fit_transform(X_train)
    X_test_scaled = abubakar_scaler.transform(X_test)

    abubakar_mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=400,
        random_state=42
    )
    abubakar_mlp.fit(X_train_scaled, y_train)

    preds = abubakar_mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    abubakar_applicant = pd.DataFrame([{
        "income": 95000.0,
        "credit_score": 780.0,
        "debt_to_income": 0.22,
        "savings": 32000.0,
        "age": 25.0
    }])
    abubakar_pred = abubakar_mlp.predict(abubakar_scaler.transform(abubakar_applicant))[0]
    tier_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
    print(f"Risk evaluation for Abubakar Iqbal: {tier_map[abubakar_pred]}")

if __name__ == "__main__":
    main()
