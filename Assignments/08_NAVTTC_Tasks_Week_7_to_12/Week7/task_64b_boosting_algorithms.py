# Abubakar Iqbal
# NAVTTC Week 7 - Task 64b: Boosting Algorithms (AdaBoost vs Gradient Boosting)

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def main():
    print("NAVTTC Week 7 Task 64b - Abubakar Iqbal")

    abubakar_X, abubakar_y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        abubakar_X, abubakar_y, test_size=0.25, random_state=42, stratify=abubakar_y
    )

    abubakar_adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=0.8, random_state=42)
    abubakar_adaboost.fit(X_train, y_train)
    ada_preds = abubakar_adaboost.predict(X_test)
    ada_probs = abubakar_adaboost.predict_proba(X_test)[:, 1]

    abubakar_gbm = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    abubakar_gbm.fit(X_train, y_train)
    gbm_preds = abubakar_gbm.predict(X_test)
    gbm_probs = abubakar_gbm.predict_proba(X_test)[:, 1]

    results = [
        {
            "Model": "AdaBoost",
            "Accuracy": f"{accuracy_score(y_test, ada_preds)*100:.2f}%",
            "ROC-AUC": f"{roc_auc_score(y_test, ada_probs):.4f}"
        },
        {
            "Model": "Gradient Boosting",
            "Accuracy": f"{accuracy_score(y_test, gbm_preds)*100:.2f}%",
            "ROC-AUC": f"{roc_auc_score(y_test, gbm_probs):.4f}"
        }
    ]
    print("\nComparison Results:")
    print(pd.DataFrame(results).to_string(index=False))

    print("\nGradient Boosting Report:")
    print(classification_report(y_test, gbm_preds, zero_division=0))

if __name__ == "__main__":
    main()
