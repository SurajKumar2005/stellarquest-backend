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
# Load Models
# ----------------------------

reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")


# ----------------------------
# FastAPI App
# ----------------------------

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Stellar Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    input_dict = data.dict()

    # 🚨 Safety Check: Prevent completely empty input
    if all(v is None for v in input_dict.values()):
        raise HTTPException(
            status_code=400,
            detail="At least one feature must be provided for prediction."
        )

    try:

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Fill missing values with np.nan so the model's pipeline imputer can handle them
        input_df = input_df.fillna(np.nan)

        # Ensure correct feature order
        input_df = input_df[reg_model.feature_names_in_]

        # -----------------------
        # Regression Prediction
        # -----------------------

        pred_log = reg_model.predict(input_df)

        predicted_radius = float(np.expm1(pred_log)[0])


        # -----------------------
        # Classification
        # -----------------------

        class_pred = int(clf_model.predict(input_df)[0])

        probability = float(clf_model.predict_proba(input_df)[0][1])


        # Convert label
        if class_pred == 1:
            label = "Confirmed"
        else:
            label = "False Positive"

        # -----------------------
        # Database Insertion
        # -----------------------

        db_record = models.PredictionHistory(
            koi_period=input_dict.get("koi_period"),
            koi_duration=input_dict.get("koi_duration"),
            koi_depth=input_dict.get("koi_depth"),
            koi_impact=input_dict.get("koi_impact"),
            koi_model_snr=input_dict.get("koi_model_snr"),
            koi_num_transits=input_dict.get("koi_num_transits"),
            koi_ror=input_dict.get("koi_ror"),
            st_teff=input_dict.get("st_teff"),
            st_logg=input_dict.get("st_logg"),
            st_met=input_dict.get("st_met"),
            st_mass=input_dict.get("st_mass"),
            st_radius=input_dict.get("st_radius"),
            st_dens=input_dict.get("st_dens"),
            teff_err1=input_dict.get("teff_err1"),
            teff_err2=input_dict.get("teff_err2"),
            logg_err1=input_dict.get("logg_err1"),
            logg_err2=input_dict.get("logg_err2"),
            feh_err1=input_dict.get("feh_err1"),
            feh_err2=input_dict.get("feh_err2"),
            mass_err1=input_dict.get("mass_err1"),
            mass_err2=input_dict.get("mass_err2"),
            radius_err1=input_dict.get("radius_err1"),
            radius_err2=input_dict.get("radius_err2"),
            predicted_radius=float(predicted_radius),
            habitability_class=label,
            confidence=float(probability)
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

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ----------------------------
# History Endpoint
# ----------------------------

@app.get("/history")
def get_prediction_history(page: int = 1, limit: int = 5, db: Session = Depends(database.get_db)):
    offset = (page - 1) * limit
    
    # Get total count for frontend pagination math
    total_count = db.query(models.PredictionHistory).count()
    
    # Query descending rows with offset
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

# Serve static assets (JS, CSS) from the 'assets' folder
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Serve the index.html for the root route
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")
    
@app.get("/{catchall:path}")
async def serve_spa(catchall: str):
    """Fallback route for SPA navigation"""
    if os.path.isfile(catchall):
        return FileResponse(catchall)
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)