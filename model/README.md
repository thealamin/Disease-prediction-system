# 🩺 Disease Prediction Web App

A symptom-based disease predictor for everyday people.
Enter your symptoms → get a likely condition + the right doctor to visit.

---

## 📁 Project Structure

```
disease_predictor/
├── model/
│   ├── train.py          ← Train the ML model (run this first)
│   ├── predictor.py      ← Prediction logic (used by the API)
│   ├── doctor_map.py     ← Disease → Doctor type mapping
│   ├── dataset.csv       ← ⬇️  You download this from Kaggle
│   ├── trained_model.pkl ← Auto-generated after training
│   └── symptom_list.pkl  ← Auto-generated after training
├── api/
│   └── main.py           ← FastAPI backend (coming next)
├── frontend/
│   └── ...               ← React app (coming after API)
└── requirements.txt
```

---

## 🚀 Setup Guide

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset
1. Go to: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
2. Download `dataset.csv`
3. Place it inside the `model/` folder

### Step 3 — Train the model
```bash
cd model
python train.py
```
This will create `trained_model.pkl` and `symptom_list.pkl` automatically.

### Step 4 — Run the API (coming next)
```bash
cd api
uvicorn main:app --reload
```

### Step 5 — Start the frontend (coming after API)
```bash
cd frontend
npm install
npm start
```

---

## ⚠️ Disclaimer
This app is **not a medical diagnosis tool**.
It is intended to help everyday people understand their symptoms
and find the right type of doctor. Always consult a real doctor.
