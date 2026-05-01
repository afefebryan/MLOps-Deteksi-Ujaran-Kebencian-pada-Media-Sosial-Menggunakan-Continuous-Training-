import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# KONFIGURASI
REGISTRY_NAME = "HateSpeechClassifierV1"
MLRUNS_PATH   = "mlruns"

mlflow.set_tracking_uri(MLRUNS_PATH)
client = MlflowClient()

# STEP 1 — Promosikan model dari Staging ke Production
def promote_to_production(registry_name: str):
    # Cek apakah sudah ada di Production
    production_versions = client.get_latest_versions(registry_name, stages=["Production"])
    if production_versions:
        latest = production_versions[0]
        print(f"Model sudah berada di Production:")
        print(f"  Nama     : {latest.name}")
        print(f"  Versi    : {latest.version}")
        print(f"  Deskripsi: {latest.description}\n")
        return latest.version

    # Kalau belum, cek Staging lalu promosikan
    staging_versions = client.get_latest_versions(registry_name, stages=["Staging"])
    if not staging_versions:
        print(f"Tidak ada model di Staging maupun Production untuk '{registry_name}'.")
        return None

    latest = staging_versions[0]
    print(f"Model ditemukan di Staging, mempromosikan ke Production...")
    print(f"  Nama     : {latest.name}")
    print(f"  Versi    : {latest.version}")
    print(f"  Deskripsi: {latest.description}")

    client.transition_model_version_stage(
        name                      = registry_name,
        version                   = latest.version,
        stage                     = "Production",
        archive_existing_versions = True
    )
    print(f"  -> Model v{latest.version} berhasil dipromosikan ke Production\n")
    return latest.version


# STEP 2 — Load model dari Production
def load_production_model(registry_name: str):
    model_uri = f"models:/{registry_name}/Production"
    print(f"Memuat model dari : {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"  Model berhasil dimuat: {type(model)}\n")
    return model


# STEP 3 — Simulasi inferensi
def run_inference(model):
    sample_inputs = pd.DataFrame({
        "clean_tweet": [
            "i hate all of you so much",
            "have a great day everyone",
            "you are the worst person alive",
            "let us work together for a better future",
            "kill yourself nobody likes you",
        ]
    })

    print("Input sampel:")
    print(sample_inputs.to_string(index=False))
    print()

    import numpy as np

    texts = sample_inputs["clean_tweet"].tolist()

    # pyfunc butuh DataFrame dengan kolom clean_tweet
    preds = model.predict(pd.DataFrame({"clean_tweet": texts})).tolist()

    print(f"  preds : {preds}")
    print(f"  len   : {len(preds)}")
    print()

    label_map = {0: "neutral", 1: "harmful"}
    results   = pd.DataFrame({
        "clean_tweet": texts,
        "prediction" : preds,
        "label"      : [label_map.get(p, str(p)) for p in preds]
    })

    print("Hasil prediksi:")
    print(results.to_string(index=False))
    return results


# STEP 4 — Validasi model info di registry
def print_model_info(registry_name: str):
    print("\n" + "=" * 60)
    print("INFO MODEL DI REGISTRY")
    print("=" * 60)

    for stage in ["Production", "Staging", "Archived"]:
        versions = client.get_latest_versions(registry_name, stages=[stage])
        for v in versions:
            print(f"  Versi {v.version} | Stage: {v.current_stage}")
            print(f"  Run ID     : {v.run_id}")
            print(f"  Deskripsi  : {v.description}")
            print()


# MAIN
if __name__ == "__main__":
    print("=" * 60)
    print("SIMULASI INFERENSI — MLflow Model Registry")
    print("=" * 60 + "\n")

    # Promosikan Staging -> Production
    version = promote_to_production(REGISTRY_NAME)

    if version:
        # Load dari Production
        model = load_production_model(REGISTRY_NAME)

        # Jalankan inferensi
        print("=" * 60)
        print("HASIL INFERENSI")
        print("=" * 60)
        run_inference(model)

        # Tampilkan info registry
        print_model_info(REGISTRY_NAME)

        print("=" * 60)
        print("Simulasi inferensi selesai.")
        print("Cek MLflow UI : mlflow ui  ->  http://localhost:5000")
        print("=" * 60)
    else:
        print("Tidak ada model yang bisa dimuat. Pastikan train.py sudah dijalankan.")