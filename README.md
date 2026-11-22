# 🩺 লক্ষণ বিশ্লেষক · Disease Prediction System

A symptom checker for everyday people. Pick your symptoms in a simple Bengali interface and get the most likely condition, the top three matches with confidence scores, and the type of doctor to visit.

> ⚠️ This is a learning and awareness project, **not a medical diagnosis tool**. Always consult a qualified doctor.

## How it works

1. **Model:** a Random Forest classifier (scikit-learn, 200 trees) trained on a public symptom–disease dataset
2. **API:** a FastAPI backend with two endpoints: `GET /symptoms` lists every symptom the model knows, and `POST /predict` returns the predicted condition, the top three matches and the recommended specialist
3. **Frontend:** a single-page Bengali interface (HTML, CSS and JavaScript) that talks to the API

## Project structure

```
api/        FastAPI app (main.py)
model/      training script, predictor, disease → doctor mapping, trained model
frontend/   Bengali web interface (index.html)
```

## Run it locally

```bash
# 1. Install dependencies
pip install -r model/requirements.txt

# 2. (Optional) Retrain the model
cd model && python train.py && cd ..

# 3. Start the API at http://localhost:8000 (docs at /docs)
cd api && uvicorn main:app --reload
```

Then open `frontend/index.html` in your browser.

## Tech

Python · scikit-learn · pandas · FastAPI · Uvicorn · HTML, CSS and JavaScript

---

Made by [Alamin](https://thealamin.com), website designer, developer and cyber security expert. Need a web app, dashboard or custom tool built? [Let's talk](https://thealamin.com/contact.html).
