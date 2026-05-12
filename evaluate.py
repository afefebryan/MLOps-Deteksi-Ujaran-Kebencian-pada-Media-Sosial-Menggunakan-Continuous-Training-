"""
evaluate.py
Membandingkan metrik model terbaru di MLflow dengan threshold di params.yaml.
Exit code 0 = lolos, Exit code 1 = gagal (GitHub Actions akan stop pipeline).

Jalankan: python evaluate.py
"""

import sys
import yaml
import mlflow
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------
# LOAD KONFIGURASI
# ---------------------------------------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

F1_THRESHOLD  = params["evaluation"]["f1_macro_threshold"]
ACC_THRESHOLD = params["evaluation"]["accuracy_threshold"]
EXPERIMENT    = params["experiment"]["name"]
REGISTRY_NAME = params["experiment"]["registry_name"]

mlflow.set_tracking_uri("mlruns")
client = MlflowClient()

# ---------------------------------------------------------------
# AMBIL RUN TERBARU DARI EXPERIMENT
# ---------------------------------------------------------------
print("=" * 60)
print("MODEL EVALUATION & VALIDATION")
print("=" * 60)

experiment = client.get_experiment_by_name(EXPERIMENT)
if experiment is None:
    print(f"Experiment '{EXPERIMENT}' tidak ditemukan. Jalankan train.py dulu.")
    sys.exit(1)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_macro DESC"],
    max_results=1
)

if not runs:
    print("Tidak ada run yang ditemukan.")
    sys.exit(1)

best_run    = runs[0]
run_id      = best_run.info.run_id
f1_macro    = best_run.data.metrics.get("f1_macro", 0)
accuracy    = best_run.data.metrics.get("accuracy", 0)
clf_name    = best_run.data.tags.get("model_type", "unknown")
run_name    = best_run.info.run_name

print(f"\nRun terbaik    : {run_name}")
print(f"Run ID         : {run_id}")
print(f"Classifier     : {clf_name}")
print(f"f1_macro       : {f1_macro:.4f}  (threshold >= {F1_THRESHOLD})")
print(f"accuracy       : {accuracy:.4f}  (threshold >= {ACC_THRESHOLD})")

# ---------------------------------------------------------------
# VALIDASI THRESHOLD
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("HASIL VALIDASI")
print("=" * 60)

passed = True

if f1_macro >= F1_THRESHOLD:
    print(f"  [PASS] f1_macro  : {f1_macro:.4f} >= {F1_THRESHOLD}")
else:
    print(f"  [FAIL] f1_macro  : {f1_macro:.4f} < {F1_THRESHOLD}")
    passed = False

if accuracy >= ACC_THRESHOLD:
    print(f"  [PASS] accuracy  : {accuracy:.4f} >= {ACC_THRESHOLD}")
else:
    print(f"  [FAIL] accuracy  : {accuracy:.4f} < {ACC_THRESHOLD}")
    passed = False

# ---------------------------------------------------------------
# REGISTER KE STAGING JIKA LOLOS
# ---------------------------------------------------------------
if passed:
    print("\nValidasi LOLOS — mendaftarkan model ke Staging ...")

    artifact_path = run_name
    model_uri     = f"runs:/{run_id}/{artifact_path}"

    registered = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)

    client.update_model_version(
        name        = REGISTRY_NAME,
        version     = registered.version,
        description = (
            f"Auto-registered via evaluate.py | "
            f"run: {run_name} | "
            f"f1_macro: {f1_macro:.4f} | "
            f"accuracy: {accuracy:.4f}"
        )
    )
    client.transition_model_version_stage(
        name                      = REGISTRY_NAME,
        version                   = registered.version,
        stage                     = "Staging",
        archive_existing_versions = True
    )

    print(f"  Model '{REGISTRY_NAME}' v{registered.version} -> Staging")
    print(f"  URI : {model_uri}")
    print("\nEvaluasi selesai. Pipeline berhasil.")
    sys.exit(0)

else:
    print("\nValidasi GAGAL — model tidak didaftarkan.")
    print("Periksa hasil training dan tuning parameter.")
    sys.exit(1)