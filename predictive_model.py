import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# Load Dataset
df = pd.read_csv("train.csv")

print("Dataset Loaded Successfully!\n")

# Select Required Columns
data = df[['Pclass', 'Sex', 'Age', 'Fare', 'Survived']].copy()

# Handle Missing Values
data['Age'] = data['Age'].fillna(data['Age'].median())

# Convert Categorical Data to Numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Features and Target
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("Confusion Matrix saved as confusion_matrix.png")

# -----------------------------
# ROC Curve
# -----------------------------

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print(f"ROC AUC Score: {roc_auc:.2f}")
print("ROC Curve saved as roc_curve.png")

# -----------------------------
# Feature Importance
# -----------------------------

importance = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:\n")
print(importance_df)

plt.figure(figsize=(6, 4))
plt.bar(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("Feature Importance graph saved as feature_importance.png")

print("\nProject Execution Completed Successfully!")