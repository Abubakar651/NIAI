# Task 53 - Decision Tree Classification with Scikit-Learn
# Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "glucose": [85, 120, 140, 90, 160, 110, 130, 95, 175, 105, 150, 80],
    "blood_pressure": [66, 80, 85, 68, 90, 72, 82, 70, 92, 74, 86, 64],
    "insulin": [29, 105, 180, 35, 230, 85, 130, 40, 280, 78, 190, 25],
    "bmi": [26.6, 32.0, 33.6, 28.1, 35.3, 27.5, 30.5, 25.4, 38.0, 29.0, 34.2, 24.0],
    "age": [31, 45, 52, 26, 54, 30, 42, 28, 59, 33, 49, 23],
    "outcome": [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
X = df[["glucose", "blood_pressure", "insulin", "bmi", "age"]]
y = df["outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual outcomes:   ", list(y_test))
print("Predicted outcomes:", list(y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score for Abubakar's Decision Tree: {acc * 100:.1f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
