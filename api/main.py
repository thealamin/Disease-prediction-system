"""
Disease Prediction API
======================
Run with:  uvicorn main:app --reload  (from the /api folder)
Docs at:   http://localhost:8000/docs
"""

import sys
import os
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Point to the model folder ────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
sys.path.insert(0, MODEL_DIR)

from doctor_map import get_doctor  # noqa: E402

# ── Load model & symptom list once at startup ────────────────────────────────
with open(os.path.join(MODEL_DIR, "trained_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "symptom_list.pkl"), "rb") as f:
    ALL_SYMPTOMS: list[str] = pickle.load(f)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Disease Prediction API",
    description="Enter symptoms, get a predicted condition and doctor recommendation.",
    version="1.0.0",
)

# Allow the React frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    symptoms: list[str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"symptoms": ["fever", "headache", "nausea", "vomiting"]}
            ]
        }
    }

class DiseaseResult(BaseModel):
    disease: str
    confidence: float
    doctor: str

class PredictResponse(BaseModel):
    prediction: DiseaseResult
    top3: list[DiseaseResult]
    disclaimer: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Disease Prediction API is running 🩺"}


@app.get("/symptoms", response_model=list[str])
def get_symptoms():
    """Return the full list of symptoms the model understands."""
    return sorted(ALL_SYMPTOMS)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Given a list of symptoms, return:
    - The most likely disease
    - Confidence percentage
    - Recommended doctor type
    - Top 3 predictions
    """
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Please provide at least one symptom.")

    if len(request.symptoms) > 17:
        raise HTTPException(status_code=400, detail="Maximum 17 symptoms allowed.")

    # Normalise symptoms to lowercase + strip
    user_set = {s.strip().lower() for s in request.symptoms}

    # Build feature vector (1 = symptom present, 0 = not)
    features = [
        1 if sym.lower() in user_set else 0
        for sym in ALL_SYMPTOMS
    ]

    # Get probabilities for all diseases
    proba = model.predict_proba([features])[0]
    classes = model.classes_

    # Sort by confidence descending
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    # Build top 3 results
    top3 = [
        DiseaseResult(
            disease=disease,
            confidence=round(conf * 100, 1),
            doctor=get_doctor(disease),
        )
        for disease, conf in ranked[:3]
    ]

    return PredictResponse(
        prediction=top3[0],
        top3=top3,
        disclaimer=(
            "This is not a medical diagnosis. "
            "Please consult a qualified doctor for proper evaluation and treatment."
        ),
    )
