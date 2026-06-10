import os
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import uvicorn

from prometheus_client import Gauge

MODEL_F1_MACRO = Gauge(
    "model_f1_macro",
    "Current model F1 Macro"
)

MODEL_F1_MACRO.set(0.85)
# ---------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------
REGISTRY_NAME = os.getenv("REGISTRY_NAME", "HateSpeechClassifierV1")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

app = FastAPI(
    title       = "Hate Speech Detection API",
    description = "Inferensi model deteksi ujaran kebencian via MLflow Model Registry",
    version     = "1.0.0"
)

# ---------------------------------------------------------------
# PROMETHEUS METRICS KUSTOM
# ---------------------------------------------------------------
PREDICTION_COUNTER = Counter(
    "hate_speech_predictions_total",
    "Total prediksi yang dilakukan",
    ["label"]
)

INFERENCE_LATENCY = Histogram(
    "hate_speech_inference_latency_seconds",
    "Latensi inferensi model dalam detik",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

MODEL_VERSION_GAUGE = Gauge(
    "hate_speech_model_version",
    "Versi model yang sedang aktif"
)

REQUEST_INPUT_LENGTH = Histogram(
    "hate_speech_input_length",
    "Panjang teks input (jumlah karakter)",
    buckets=[10, 50, 100, 200, 500, 1000]
)

# ---------------------------------------------------------------
# INSTRUMENTASI OTOMATIS (latensi, throughput, status code)
# ---------------------------------------------------------------
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ---------------------------------------------------------------
# STATE
# ---------------------------------------------------------------
model         = None
model_version = None

# ---------------------------------------------------------------
# SCHEMA
# ---------------------------------------------------------------
class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    predictions: List[dict]
    model_version: str
    stage: str

# ---------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global model, model_version

    for stage in ["Production", "Staging"]:
        versions = client.get_latest_versions(REGISTRY_NAME, stages=[stage])
        if versions:
            v             = versions[0]
            model_uri     = f"models:/{REGISTRY_NAME}/{stage}"
            model         = mlflow.pyfunc.load_model(model_uri)
            model_version = v.version
            MODEL_VERSION_GAUGE.set(float(v.version))
            print(f"Model dimuat: {REGISTRY_NAME} v{v.version} ({stage})")
            return

    raise RuntimeError(
        f"Tidak ada model di Production/Staging untuk '{REGISTRY_NAME}'."
    )

# ---------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status"       : "ok",
        "model_loaded" : model is not None,
        "registry_name": REGISTRY_NAME,
        "model_version": model_version,
        "mlflow_uri"   : MLFLOW_URI,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat.")
    if not request.texts:
        raise HTTPException(status_code=400, detail="Input texts tidak boleh kosong.")

    # catat panjang input
    for text in request.texts:
        REQUEST_INPUT_LENGTH.observe(len(text))

    # ukur latensi inferensi
    start = time.time()
    df    = pd.DataFrame({"clean_tweet": request.texts})
    preds = model.predict(df)
    INFERENCE_LATENCY.observe(time.time() - start)

    label_map = {0: "neutral", 1: "harmful"}
    results   = []
    for text, pred in zip(request.texts, preds):
        label = label_map.get(int(pred), str(pred))
        PREDICTION_COUNTER.labels(label=label).inc()
        results.append({
            "text"      : text,
            "prediction": int(pred),
            "label"     : label
        })

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

@app.get("/simulate_drift")
def simulate_drift():
    MODEL_F1_MACRO.set(0.72)
    return {"f1_macro": 0.72}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)