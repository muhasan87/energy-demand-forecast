from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(
    title="NSW Energy Demand Forecasting API",
    description="Predicts hourly electricity demand for NSW Australia using weather conditions",
    version="1.0.0",
)

# ── Load model once when API starts ───────────────────────
with open("src/models/saved/xgboost.pkl", "rb") as f:
    model = pickle.load(f)


# ── Define input schema ───────────────────────────────────
class PredictionInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    cloud_cover: float
    apparent_temp: float
    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    is_peak_hour: int
    is_holiday: int
    is_extreme_heat: int
    season: int
    is_business_day: int
    demand_1hr_ago: float
    demand_24hr_ago: float
    demand_168hr_ago: float
    temp_1hr_ago: float
    temp_change: float


# ── Health check endpoint ─────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "version": "1.0.0"}


# ── Prediction endpoint ───────────────────────────────────
@app.post("/predict")
def predict(data: PredictionInput):
    features = pd.DataFrame(
        [
            {
                "temperature": data.temperature,
                "humidity": data.humidity,
                "wind_speed": data.wind_speed,
                "cloud_cover": data.cloud_cover,
                "apparent_temp": data.apparent_temp,
                "hour": data.hour,
                "day_of_week": data.day_of_week,
                "month": data.month,
                "is_weekend": data.is_weekend,
                "is_peak_hour": data.is_peak_hour,
                "is_holiday": data.is_holiday,
                "is_extreme_heat": data.is_extreme_heat,
                "season": data.season,
                "is_business_day": data.is_business_day,
                "demand_1hr_ago": data.demand_1hr_ago,
                "demand_24hr_ago": data.demand_24hr_ago,
                "demand_168hr_ago": data.demand_168hr_ago,
                "temp_1hr_ago": data.temp_1hr_ago,
                "temp_change": data.temp_change,
            }
        ]
    )

    prediction = model.predict(features)[0]

    return {
        "predicted_demand_mw": round(float(prediction), 2),
        "unit": "MW",
        "region": "NSW1",
        "model": "XGBoost",
    }


# ── Model info endpoint ───────────────────────────────────
@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost",
        "features": 19,
        "training_data": "AEMO NSW 2025 (8,592 hourly records)",
        "performance": {"RMSE": "259.37 MW", "MAE": "180.16 MW", "MAPE": "2.79%"},
    }
