import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================
# Load Synthetic Dataset
# =========================================

data_path = "../dataset/credit_risk.csv"
df = pd.read_csv(data_path)

print("✅ Dataset Loaded:", df.shape)

# =========================================
# Separate Target
# =========================================

X = df.drop("Risk_Label", axis=1)
y = df["Risk_Label"]

# =========================================
# One-Hot Encode Categorical Columns
# =========================================

categorical_cols = ["Employment_Type", "Credit_History", "Loan_Purpose"]

X_encoded = pd.get_dummies(X, columns=categorical_cols)

print("✅ Categorical Encoding Done")

# Save column structure (VERY IMPORTANT for Django)
joblib.dump(X_encoded.columns.tolist(), "../saved_model/model_columns.pkl")

# =========================================
# Feature Scaling
# =========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

print("✅ Feature Scaling Completed")

# =========================================
# Train-Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("✅ Data Split Completed")

# =========================================
# Train Random Forest Model
# =========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

print("✅ Model Training Finished")

# =========================================
# Evaluate Model
# =========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================================
# Save Model Artifacts
# =========================================

os.makedirs("../saved_model", exist_ok=True)

joblib.dump(model, "../saved_model/credit_model.pkl")
joblib.dump(scaler, "../saved_model/scaler.pkl")

print("\n✅ Model Saved Successfully!")