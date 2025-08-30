# titanic.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# 1. Load dataset
data = pd.read_csv("train.csv")
print(data.head())

# 2. Data Cleaning
# Handle missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
data.drop("Cabin", axis=1, inplace=True)  # too many missing values

# Drop irrelevant columns
data.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# 3. Encode categorical values
le = LabelEncoder()
data["Sex"] = le.fit_transform(data["Sex"])        # Male=1, Female=0
data["Embarked"] = le.fit_transform(data["Embarked"])

# 4. Split features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Models
# Logistic Regression
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# 6. Predictions
log_preds = log_model.predict(X_test)
tree_preds = tree_model.predict(X_test)

# 7. Evaluation
print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, log_preds))
print(confusion_matrix(y_test, log_preds))
print(classification_report(y_test, log_preds))

print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, tree_preds))
print(confusion_matrix(y_test, tree_preds))
print(classification_report(y_test, tree_preds))

# 8. ROC Curve (for Logistic Regression)
y_probs = log_model.predict_proba(X_test)[:,1]  # Probabilities for class=1
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="ROC Curve (AUC = %.2f)" % roc_auc)
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.show()
