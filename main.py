from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
import models
import database
import pandas as pd
import numpy as np
import joblib
import os


# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(title="Stellar Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Load Models (SAFE)
# ----------------------------

try:
    reg_model = joblib.load("regression_model.pkl")
    clf_model = joblib.load("classification_model.pkl")
except Exception as e:
    print("Model loading failed:", e)
    reg_model = None
    clf_model = None


# ----------------------------
# Startup Event (DB tables)
# ----------------------------

@app.on_event("startup")
def startup():
    models.Base.metadata.create_all(bind=database.engine)


# ----------------------------
# Input Schema
# ----------------------------

class StellarInput(BaseModel):

    koi_period: Optional[float] = None
    koi_duration: Optional[float] = None
    koi_depth: Optional[float] = None
    koi_impact: Optional[float] = None
    koi_model_snr: Optional[float] = None
    koi_num_transits: Optional[float] = None
    koi_ror: Optional[float] = None
    st_teff: Optional[float] = None
    st_logg: Optional[float] = None
    st_met: Optional[float] = None
    st_mass: Optional[float] = None
    st_radius: Optional[float] = None
    st_dens: Optional[float] = None
    teff_err1: Optional[float] = None
    teff_err2: Optional[float] = None
    logg_err1: Optional[float] = None
    logg_err2: Optional[float] = None
    feh_err1: Optional[float] = None
    feh_err2: Optional[float] = None
    mass_err1: Optional[float] = None
    mass_err2: Optional[float] = None
    radius_err1: Optional[float] = None
    radius_err2: Optional[float] = None


# ----------------------------
# Prediction Endpoint
# ----------------------------

@app.post("/predict")
def predict(data: StellarInput, db: Session = Depends(database.get_db)):

    if reg_model is None or clf_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    input_dict = data.dict()

    if all(v is None for v in input_dict.values()):
        raise HTTPException(
            status_code=400,
            detail="At least one feature must be provided for prediction."
        )

    try:

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.fillna(np.nan)
        input_df = input_df[reg_model.feature_names_in_]

        pred_log = reg_model.predict(input_df)
        predicted_radius = float(np.expm1(pred_log)[0])

        class_pred = int(clf_model.predict(input_df)[0])
        probability = float(clf_model.predict_proba(input_df)[0][1])

        label = "Confirmed" if class_pred == 1 else "False Positive"

        db_record = models.PredictionHistory(
            **input_dict,
            predicted_radius=predicted_radius,
            habitability_class=label,
            confidence=probability
        )

        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return {
            "predicted_planet_radius": round(predicted_radius,4),
            "habitability_class": label,
            "habitability_probability": round(probability,4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# History Endpoint
# ----------------------------

@app.get("/history")
def get_prediction_history(page: int = 1, limit: int = 5, db: Session = Depends(database.get_db)):

    offset = (page - 1) * limit
    total_count = db.query(models.PredictionHistory).count()

    history_records = (
        db.query(models.PredictionHistory)
        .order_by(models.PredictionHistory.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "data": history_records,
        "total": total_count,
        "page": page,
        "limit": limit
    }


# ----------------------------
# Frontend Serving
# ----------------------------

app.mount("/assets", StaticFiles(directory="assets"), name="assets")

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.get("/{catchall:path}")
async def serve_spa(catchall: str):
    if os.path.isfile(catchall):
        return FileResponse(catchall)
    return FileResponse("index.html")
