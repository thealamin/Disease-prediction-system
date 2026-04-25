"""
Disease Prediction Model — Training Script
==========================================
Dataset : Kaggle "Disease Symptom Prediction" dataset
          https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
          Download dataset.csv and place it in the same folder as this script.

Output  : trained_model.pkl  — saved Random Forest classifier
          symptom_list.pkl   — ordered list of all symptom columns
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")

if not os.path.exists(DATASET_PATH):
    print("❌  dataset.csv not found!")
    print("    Please download it from:")
    print("    https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
    print("    and place it in the same folder as this script.")
    sys.exit(1)

print("📂  Loading dataset ...")
df = pd.read_csv(DATASET_PATH)
print(f"    Rows: {len(df)}  |  Columns: {len(df.columns)}")

# ── 2. CLEAN DATA ─────────────────────────────────────────────────────────────

# Strip whitespace from column names and string values
df.columns = df.columns.str.strip()
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# The dataset has: Disease | Symptom_1 | Symptom_2 | ... | Symptom_17
# We'll one-hot encode all symptoms into a single feature matrix.

print("🔧  Preprocessing ...")

disease_col = "Disease"
symptom_cols = [c for c in df.columns if c.startswith("Symptom")]

# Collect every unique symptom across all columns
all_symptoms = sorted(
    set(
        sym
        for col in symptom_cols
        for sym in df[col].dropna().unique()
    )
)
print(f"    Unique symptoms found : {len(all_symptoms)}")
print(f"    Unique diseases found : {df[disease_col].nunique()}")

# Build a binary feature matrix: 1 if the patient has that symptom, else 0
def encode_row(row):
    present = set(row[symptom_cols].dropna().values)
    return [1 if sym in present else 0 for sym in all_symptoms]

X = pd.DataFrame(
    [encode_row(row) for _, row in df.iterrows()],
    columns=all_symptoms,
)
y = df[disease_col]

# ── 3. TRAIN MODEL ────────────────────────────────────────────────────────────

print("🤖  Training Random Forest classifier ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 4. EVALUATE ───────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅  Accuracy : {acc * 100:.2f}%")
print("\n📊  Classification Report:\n")
print(classification_report(y_test, y_pred))

# ── 5. SAVE MODEL ─────────────────────────────────────────────────────────────

model_path   = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
symptom_path = os.path.join(os.path.dirname(__file__), "symptom_list.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(symptom_path, "wb") as f:
    pickle.dump(all_symptoms, f)

print(f"\n💾  Model saved  → {model_path}")
print(f"💾  Symptoms saved → {symptom_path}")
print("\n🎉  Training complete! You can now run the API.")
