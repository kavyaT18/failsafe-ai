from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import io
import os
from dotenv import load_dotenv
from utils import feature_eng, get_risk_tier, get_shap_reasons

load_dotenv()
app = FastAPI(title="FailSafe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "failsafe_model3.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded.")
    else:
        print(f"WARNING: model not found at {MODEL_PATH}")


# ── Single student prediction ──────────────────────────────────────────────

class StudentInput(BaseModel):
    sex: str; age: int; address: str; Medu: int; Fedu: int
    studytime: int; failures: int; schoolsup: str; famsup: str
    paid: str; activities: str; nursery: str; higher: str
    absences: int; G1: float; G2: float

@app.post("/predict")
def predict_single(student: StudentInput):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    df = pd.DataFrame([student.model_dump()])
    df = feature_eng(df)

    prob  = float(model.predict_proba(df)[:, 1][0])
    pred  = int(model.predict(df)[0])
    tier  = get_risk_tier(prob)
    shap_reasons = get_shap_reasons(model, df)

    llm_summary = None
    if os.getenv("GROQ_API_KEY"):
        llm_summary = get_llm_intervention(student.model_dump(), prob, tier, shap_reasons)

    return {
        "prediction":    pred,
        "risk_prob":     round(prob * 100, 1),
        "risk_tier":     tier,
        "shap_reasons":  shap_reasons,
        "llm_summary":   llm_summary,
    }


# ── Bulk CSV upload ────────────────────────────────────────────────────────

@app.post("/predict/bulk")
async def predict_bulk(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    content = await file.read()
    try:
        raw = pd.read_csv(io.BytesIO(content), sep=';')
    except Exception:
        raw = pd.read_csv(io.BytesIO(content))

    df = feature_eng(raw.copy())

    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    results = []
    for i, (prob, pred) in enumerate(zip(probs, preds)):
        results.append({
            "student_id":  int(raw.index[i]),
            "risk_prob":   round(float(prob) * 100, 1),
            "risk_tier":   get_risk_tier(float(prob)),
            "prediction":  int(pred),
            "G1":          float(raw.iloc[i].get("G1", 0)),
            "G2":          float(raw.iloc[i].get("G2", 0)),
            "absences":    int(raw.iloc[i].get("absences", 0)),
        })

    summary = {
        "total": len(results),
        "high":   sum(1 for r in results if r["risk_tier"] == "HIGH"),
        "medium": sum(1 for r in results if r["risk_tier"] == "MEDIUM"),
        "low":    sum(1 for r in results if r["risk_tier"] == "LOW"),
    }
    return {"summary": summary, "students": results}


# ── LLM Intervention (Groq) ──────────────────────────────────────────────

def get_llm_intervention(student: dict, prob: float, tier: str, reasons: list) -> str:

    try:

        from groq import Groq
        import os

        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        reason_text = "\n".join(

            f"- {r['label']}: {r['direction']} "
            f"(importance: {abs(r['shap']):.3f})"

            for r in reasons
        )

        prompt = f"""
You are an academic advisor AI.

A student has been flagged as {tier} risk
(probability: {prob*100:.1f}%)
by our ML model.

Key risk factors:
{reason_text}

Student snapshot:
- G1 = {student['G1']}
- G2 = {student['G2']}
- failures = {student['failures']}
- absences = {student['absences']}
- studytime = {student['studytime']}
- higher_edu_goal = {student['higher']}

Generate:
1. Brief student situation summary
2. Key academic concerns
3. 3 personalized intervention recommendations
4. Monitoring strategy

Keep response:
- concise
- practical
- empathetic
- faculty-oriented

Maximum 150 words.
"""

        completion = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[

                {
                    "role": "system",

                    "content":
                    "You are an educational intervention AI assistant."
                },

                {
                    "role": "user",

                    "content": prompt
                }
            ],

            temperature=0.4,

            max_tokens=300
        )

        return completion.choices[0].message.content

    except Exception as e:

        return f"LLM unavailable: {str(e)}"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
