# Week 7 - Boosting Algorithms (AdaBoost & Gradient Boosting)
# NAVTTC Outline Week 7 Day 1 / Boosting Practical
# Reference: Scikit-Learn Ensemble Methods (AdaBoostClassifier, GradientBoostingClassifier)

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

def run_boosting_benchmark():
    print("=" * 65)
    print("Week 7 Practical: Boosting Algorithms Benchmark (AdaBoost vs Gradient Boosting)")
    print("Student: Abubakar Iqbal | NAVTTC Week 7")
    print("=" * 65)

    # Imbalanced tabular classification dataset (simulating credit/customer churn)
    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=7,
        n_redundant=3,
        weights=[0.7, 0.3],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Dataset Size: {X.shape[0]} Rows | 10 Features | Class Split: 70% Non-Default / 30% Default")

    # 1. AdaBoost (Adaptive Boosting)
    adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=42)
    adaboost.fit(X_train, y_train)
    ada_preds = adaboost.predict(X_test)
    ada_probs = adaboost.predict_proba(X_test)[:, 1]
    ada_acc = accuracy_score(y_test, ada_preds)
    ada_auc = roc_auc_score(y_test, ada_probs)

    # 2. Gradient Boosting Classifier (GBM)
    gbm = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    gbm.fit(X_train, y_train)
    gbm_preds = gbm.predict(X_test)
    gbm_probs = gbm.predict_proba(X_test)[:, 1]
    gbm_acc = accuracy_score(y_test, gbm_preds)
    gbm_auc = roc_auc_score(y_test, gbm_probs)

    print("\n--- Model Benchmark Comparison ---")
    results = [
        {"Model": "AdaBoost Classifier", "Accuracy": f"{ada_acc*100:.2f}%", "ROC-AUC": f"{ada_auc:.4f}"},
        {"Model": "Gradient Boosting (GBM)", "Accuracy": f"{gbm_acc*100:.2f}%", "ROC-AUC": f"{gbm_auc:.4f}"}
    ]
    print(pd.DataFrame(results).to_string(index=False))

    print("\nDetailed Classification Report (Gradient Boosting):")
    print(classification_report(y_test, gbm_preds, target_names=["Good (0)", "Default (1)"], zero_division=0))

    # Top Feature Importances from Gradient Boosting
    feature_importances = pd.Series(gbm.feature_importances_, index=[f"Feature_{i+1}" for i in range(10)]).sort_values(ascending=False)
    print("\nTop 5 Most Influential Features (GBM Importance):")
    for feat, imp in feature_importances.head(5).items():
        print(f"  {feat}: {imp:.4f}")

if __name__ == "__main__":
    run_boosting_benchmark()
