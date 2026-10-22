# Task 57 - Multilayer Perceptron (MLP) Classification
# NAVTTC Outline Week 7 / Task 57
# Reference: https://machinelearningmastery.com/neural-networks-crash-course/

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

def run_mlp_experiment():
    print("=" * 65)
    print("Task 57: Multilayer Perceptron (MLP) Classifier with Scikit-Learn")
    print("Student: Abubakar Iqbal | NAVTTC Week 7")
    print("=" * 65)

    # Synthetic Multi-class Credit & Lending Decision Dataset
    np.random.seed(42)
    n_samples = 300

    # Features: [Income, Credit Score, Debt-to-Income, Savings, Age]
    income = np.random.uniform(20000, 120000, n_samples)
    credit_score = np.random.uniform(300, 850, n_samples)
    dti = np.random.uniform(0.1, 0.7, n_samples)
    savings = np.random.uniform(1000, 50000, n_samples)
    age = np.random.uniform(20, 65, n_samples)

    # Multi-class target: 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk
    risk_score = (credit_score / 850) * 0.4 + (savings / 50000) * 0.3 - (dti) * 0.3
    labels = np.zeros(n_samples, dtype=int)
    labels[risk_score > 0.45] = 0   # Low Risk / Prime
    labels[(risk_score <= 0.45) & (risk_score >= 0.25)] = 1 # Moderate Risk
    labels[risk_score < 0.25] = 2   # High Risk / Subprime

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "debt_to_income": dti,
        "savings": savings,
        "age": age,
        "risk_tier": labels
    })

    print(f"\nDataset Dimensions: {df.shape}")
    print("Class Distribution:\n", df["risk_tier"].value_counts().sort_index().rename({0: "0 (Low Risk)", 1: "1 (Moderate)", 2: "2 (High Risk)"}))

    X = df.drop(columns=["risk_tier"])
    y = df["risk_tier"]

    # Train/Test Split (75% Train, 25% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Normalization (Crucial for Neural Network gradient descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Construct MLP Architecture: 5 -> 32 -> 16 -> 3
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0001,          # L2 regularization
        batch_size=32,
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15
    )

    print("\nTraining MLP Architecture:")
    print("  - Input Dimension: 5 Features")
    print("  - Hidden Layer 1: 32 Neurons (ReLU)")
    print("  - Hidden Layer 2: 16 Neurons (ReLU)")
    print("  - Output Layer: 3 Neurons (Softmax)")
    print("  - Optimizer: Adam (Initial LR = 0.01)")

    mlp.fit(X_train_scaled, y_train)

    print(f"\nModel Converged in {mlp.n_iter_} Iterations.")
    print(f"Final Training Loss: {mlp.loss_:.5f}")

    # Predictions & Benchmarking
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Set Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    target_names = ["Low Risk (0)", "Moderate (1)", "High Risk (2)"]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    run_mlp_experiment()
