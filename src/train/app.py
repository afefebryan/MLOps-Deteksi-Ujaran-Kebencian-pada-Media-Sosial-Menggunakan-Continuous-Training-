import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# KONFIGURASI
REGISTRY_NAME = os.getenv("REGISTRY_NAME", "HateSpeechClassifierV1")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

app    = FastAPI(
    title       = "Hate Speech Detection API",
    description = "Inferensi model deteksi ujaran kebencian via MLflow Model Registry",
    version     = "1.0.0"
)

model       = None
model_version = None

# SCHEMA
class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    predictions: List[dict]
    model_version: str
    stage: str

# STARTUP — load model dari Production
@app.on_event("startup")
def load_model():
    global model, model_version

    # cek Production dulu, fallback ke Staging
    for stage in ["Production", "Staging"]:
        versions = client.get_latest_versions(REGISTRY_NAME, stages=[stage])
        if versions:
            v           = versions[0]
            model_uri   = f"models:/{REGISTRY_NAME}/{stage}"
            model       = mlflow.pyfunc.load_model(model_uri)
            model_version = v.version
            print(f"Model dimuat: {REGISTRY_NAME} v{v.version} ({stage})")
            return

    raise RuntimeError(
        f"Tidak ada model di Production/Staging untuk '{REGISTRY_NAME}'. "
        f"Jalankan train.py terlebih dahulu."
    )

# ENDPOINTS
@app.get("/health")
def health():
    return {
        "status"        : "ok",
        "model_loaded"  : model is not None,
        "registry_name" : REGISTRY_NAME,
        "model_version" : model_version,
        "mlflow_uri"    : MLFLOW_URI,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")

    if not request.texts:
        raise HTTPException(status_code=400, detail="Input texts tidak boleh kosong.")

    df    = pd.DataFrame({"clean_tweet": request.texts})
    preds = model.predict(df)

    label_map = {0: "neutral", 1: "harmful"}
    results   = [
        {
            "text"      : text,
            "prediction": int(pred),
            "label"     : label_map.get(int(pred), str(pred))
        }
        for text, pred in zip(request.texts, preds)
    ]

    # ambil stage aktif
    active_stage = "Unknown"
    for stage in ["Production", "Staging"]:
        if client.get_latest_versions(REGISTRY_NAME, stages=[stage]):
            active_stage = stage
            break

    return PredictResponse(
        predictions   = results,
        model_version = str(model_version),
        stage         = active_stage
    )


@app.get("/model/info")
def model_info():
    info = {}
    for stage in ["Production", "Staging", "Archived"]:
        versions = client.get_latest_versions(REGISTRY_NAME, stages=[stage])
        if versions:
            v = versions[0]
            info[stage] = {
                "version"    : v.version,
                "run_id"     : v.run_id,
                "description": v.description,
            }
    return {"registry": REGISTRY_NAME, "versions": info}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)