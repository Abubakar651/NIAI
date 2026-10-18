# Task 54 - Support Vector Machine (SVM) Classification with Scikit-Learn
# Reference: https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "feature_1": [2.5, 1.5, 3.2, 1.8, 5.5, 6.1, 5.8, 6.5, 2.0, 5.9, 1.2, 6.8],
    "feature_2": [2.1, 1.8, 2.9, 1.2, 5.0, 5.8, 6.2, 6.9, 1.9, 5.4, 1.1, 7.1],
    "class_label": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[["feature_1", "feature_2"]]
y = df["class_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual classes:   ", list(y_test))
print("Predicted classes:", list(y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score for Abubakar's SVM Model: {acc * 100:.1f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
