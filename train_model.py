import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# ===============================
# 1. Load Dataset
# ===============================

dataset_path = "dataset/creditcard.csv"

if not os.path.exists(dataset_path):
    print("❌ Dataset not found!")
    print("Make sure creditcard.csv is inside dataset folder.")
    exit()

print("✅ Loading dataset...")
data = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully!")
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
    stratify=y   # Important for imbalanced data
)

# ===============================
# 4. Train Model
# ===============================

print("🚀 Training Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"   # Handles imbalance
)

model.fit(X_train, y_train)

print("✅ Model Training Completed!")

# ===============================
# 5. Evaluate Model
# ===============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n📊 Model Evaluation")
print("----------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# 6. Save Model
# ===============================

joblib.dump(model, "model.pkl")

print("\n💾 Model saved successfully as model.pkl")
