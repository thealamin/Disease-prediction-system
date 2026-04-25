"""
predictor.py
------------
Loads the trained model and exposes a single predict() function
that the FastAPI backend will call.
"""

import os
import pickle
from doctor_map import get_doctor

_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
_SYMPTOM_PATH = os.path.join(os.path.dirname(__file__), "symptom_list.pkl")

# Load once when the module is imported (fast for every request after that)
with open(_MODEL_PATH, "rb") as f:
    _model = pickle.load(f)

with open(_SYMPTOM_PATH, "rb") as f:
    _all_symptoms: list[str] = pickle.load(f)


def get_all_symptoms() -> list[str]:
    """Return the full list of symptoms the model understands."""
    return _all_symptoms


def predict(user_symptoms: list[str]) -> dict:
    """
    Given a list of symptom strings the user selected,
    return the predicted disease, confidence, and recommended doctor.

    Example
    -------
    predict(["fever", "headache", "nausea"])
    → {
        "disease":    "Typhoid",
        "confidence": 87,
        "doctor":     "Infectious Disease Specialist",
        "top3": [
            {"disease": "Typhoid",     "confidence": 87},
            {"disease": "Dengue",      "confidence": 9},
            {"disease": "Common Cold", "confidence": 4},
        ]
      }
    """
    # Normalise: lowercase + strip, same as training
    user_set = {s.strip().lower() for s in user_symptoms}

    # Build feature vector
    features = [
        1 if sym.lower() in user_set else 0
        for sym in _all_symptoms
    ]

    # Get class probabilities
    proba = _model.predict_proba([features])[0]
    classes = _model.classes_

    # Sort by probability descending
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    top_disease, top_conf = ranked[0]
    top3 = [
        {"disease": d, "confidence": round(c * 100, 1)}
        for d, c in ranked[:3]
    ]

    return {
        "disease":    top_disease,
        "confidence": round(top_conf * 100, 1),
        "doctor":     get_doctor(top_disease),
        "top3":       top3,
    }
