import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import joblib
import os

# ===============================
# 1. Load Dataset
# ===============================

dataset_path = "dataset/creditcard.csv"

if not os.path.exists(dataset_path):
    print("❌ Dataset not found!")
    exit()

print("✅ Loading dataset...")
data = pd.read_csv(dataset_path)

print("Total Transactions:", len(data))
print("Fraud Cases:", data["Class"].sum())

# ===============================
# 2. Split Features & Target
# ===============================

X = data.drop("Class", axis=1)
y = data["Class"]

# ===============================
# 3. Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 4. Train Random Forest
# ===============================

print("🚀 Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_probs)

print("\nRandom Forest Accuracy:", round(rf_accuracy * 100, 2), "%")
print("Random Forest AUC:", round(rf_auc, 4))


# ===============================
# 5. Train Logistic Regression
# ===============================

print("\n🚀 Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_probs)

print("\nLogistic Regression Accuracy:", round(lr_accuracy * 100, 2), "%")
print("Logistic Regression AUC:", round(lr_auc, 4))


# ===============================
# 6. Save Models
# ===============================

joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(lr_model, "lr_model.pkl")

print("\n💾 Models saved as rf_model.pkl and lr_model.pkl")
