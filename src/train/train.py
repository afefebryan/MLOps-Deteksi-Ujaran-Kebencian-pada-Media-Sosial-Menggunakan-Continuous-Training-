import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

# KONFIGURASI
DATA_PATH        = "../../data/processed/tdavidson_hate_speech_v0_clean.csv"
DATASET_NAME     = "tdavidson_hate_speech_v0_clean"
DATASET_VERSION  = "v0"
EXPERIMENT_NAME  = "hate-speech-classification"
REGISTRY_NAME    = "HateSpeechClassifier"
BEST_METRIC      = "f1_macro"

TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)

# SETUP MLFLOW
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

# LOAD & SPLIT DATA
print("Memuat dataset ...")
df = pd.read_csv(DATA_PATH)
X  = df["clean_tweet"]
y  = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train : {len(X_train)} | Test : {len(X_test)}")
print(f"Label : {dict(y.value_counts())}\n")

# HELPER
def compute_metrics(y_true, y_pred):
    return {
        "accuracy"        : accuracy_score(y_true, y_pred),
        "f1_macro"        : f1_score(y_true, y_pred, average="macro"),
        "f1_weighted"     : f1_score(y_true, y_pred, average="weighted"),
        "precision_macro" : precision_score(y_true, y_pred, average="macro"),
        "recall_macro"    : recall_score(y_true, y_pred, average="macro"),
    }


def run_experiment(run_name: str, tfidf_params: dict, clf_name: str, clf):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   clf)
    ])

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", clf_name)
        mlflow.set_tag("phase",      run_name.split("_")[0])

        # Log dataset resmi — muncul di kolom Dataset di MLflow UI
        train_df = X_train.to_frame().assign(label=y_train)
        test_df  = X_test.to_frame().assign(label=y_test)
        mlflow.log_input(
            mlflow.data.from_pandas(
                train_df, source=DATA_PATH,
                name=f"{DATASET_NAME}_train", targets="label"
            ),
            context="training"
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                test_df, source=DATA_PATH,
                name=f"{DATASET_NAME}_test", targets="label"
            ),
            context="validation"
        )

        mlflow.log_param("data_train_size",   len(X_train))
        mlflow.log_param("data_test_size",    len(X_test))
        mlflow.log_param("data_test_ratio",   0.3)
        mlflow.log_param("data_random_state", 42)

        # TF-IDF params
        mlflow.log_param("tfidf_ngram_range",  str(tfidf_params.get("ngram_range")))
        mlflow.log_param("tfidf_min_df",       tfidf_params.get("min_df"))
        mlflow.log_param("tfidf_max_df",       tfidf_params.get("max_df"))
        mlflow.log_param("tfidf_sublinear_tf", tfidf_params.get("sublinear_tf"))

        # Classifier params
        for k, v in clf.get_params().items():
            mlflow.log_param(f"clf_{k}", v)

        # Train
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Metrics
        metrics = compute_metrics(y_test, preds)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="f1_macro", n_jobs=-1
        )
        mlflow.log_metric("cv_f1_macro_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_macro_std",  cv_scores.std())

        # Artifact: classification report
        report      = classification_report(y_test, preds, target_names=["neutral", "harmful"])
        report_path = f"/tmp/{run_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path="reports")

        signature = infer_signature(X_train, preds)
        mlflow.sklearn.log_model(
            sk_model      = pipeline,
            artifact_path = run_name,
            signature     = signature,
            input_example = X_test.head(3).to_frame()
        )

        run_id = run.info.run_id
        print(f"  {run_name:<40} | acc={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f}")

    return run_id, metrics


# PHASE 1 - BASELINE EXPERIMENTS
print("=" * 60)
print("PHASE 1 - Baseline Experiments")
print("=" * 60)

baseline_configs = [
    ("Phase1_LR",
     "LogisticRegression",
     LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),

    ("Phase1_LinearSVM",
     "LinearSVC",
     LinearSVC(C=0.5, class_weight="balanced", max_iter=2000)),

    ("Phase1_SGD_SVM",
     "SGDClassifier",
     SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4,
                   max_iter=2000, class_weight="balanced", random_state=42)),

    ("Phase1_SGD_LR",
     "SGDClassifier",
     SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                   max_iter=2000, class_weight="balanced", random_state=42)),

    ("Phase1_RF",
     "RandomForestClassifier",
     RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
]

phase1_results = []

for run_name, clf_name, clf in baseline_configs:
    run_id, metrics = run_experiment(run_name, TFIDF_PARAMS, clf_name, clf)
    phase1_results.append({
        "run_id"  : run_id,
        "run_name": run_name,
        "clf_name": clf_name,
        **metrics
    })

phase1_df   = pd.DataFrame(phase1_results).sort_values(BEST_METRIC, ascending=False)
best_phase1 = phase1_df.iloc[0]

print(f"\nModel terbaik Phase 1 : {best_phase1['run_name']}")
print(f"Classifier            : {best_phase1['clf_name']}")
print(f"{BEST_METRIC}         : {best_phase1[BEST_METRIC]:.4f}\n")


# PHASE 2 - HYPERPARAMETER TUNING (model terbaik dari Phase 1)
print("=" * 60)
print(f"PHASE 2 - Hyperparameter Tuning ({best_phase1['clf_name']})")
print("=" * 60)

TUNING_GRIDS = {
    "LogisticRegression": [
        dict(C=0.1,  solver="lbfgs", max_iter=1000, class_weight="balanced"),
        dict(C=1.0,  solver="lbfgs", max_iter=1000, class_weight="balanced"),
        dict(C=5.0,  solver="lbfgs", max_iter=1000, class_weight="balanced"),
        dict(C=10.0, solver="saga",  max_iter=2000, class_weight="balanced"),
        dict(C=1.0,  solver="saga",  max_iter=1000, class_weight="balanced", penalty="l1"),
    ],
    "LinearSVC": [
        dict(C=0.1, class_weight="balanced", max_iter=3000),
        dict(C=0.5, class_weight="balanced", max_iter=3000),
        dict(C=1.0, class_weight="balanced", max_iter=3000),
        dict(C=2.0, class_weight="balanced", max_iter=3000),
        dict(C=5.0, class_weight="balanced", max_iter=3000),
    ],
    "SGDClassifier": [
        dict(loss="hinge",          penalty="l2", alpha=1e-3, max_iter=2000,
             class_weight="balanced", random_state=42),
        dict(loss="hinge",          penalty="l2", alpha=1e-4, max_iter=2000,
             class_weight="balanced", random_state=42),
        dict(loss="log_loss",       penalty="l2", alpha=1e-3, max_iter=2000,
             class_weight="balanced", random_state=42),
        dict(loss="log_loss",       penalty="l1", alpha=1e-4, max_iter=2000,
             class_weight="balanced", random_state=42),
        dict(loss="modified_huber", penalty="l2", alpha=1e-4, max_iter=2000,
             class_weight="balanced", random_state=42),
    ],
    "RandomForestClassifier": [
        dict(n_estimators=50,  max_depth=None, min_samples_split=2,
             class_weight="balanced", random_state=42),
        dict(n_estimators=100, max_depth=None, min_samples_split=2,
             class_weight="balanced", random_state=42),
        dict(n_estimators=200, max_depth=20,   min_samples_split=5,
             class_weight="balanced", random_state=42),
        dict(n_estimators=300, max_depth=None, min_samples_split=2,
             class_weight="balanced", random_state=42),
        dict(n_estimators=100, max_depth=10,   min_samples_split=10,
             class_weight="balanced", random_state=42),
    ],
}

CLF_MAP = {
    "LogisticRegression"    : LogisticRegression,
    "LinearSVC"             : LinearSVC,
    "SGDClassifier"         : SGDClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}

best_clf_name = best_phase1["clf_name"]
CLFClass      = CLF_MAP[best_clf_name]
param_grid    = TUNING_GRIDS.get(best_clf_name, [])

phase2_results = []

for i, clf_params in enumerate(param_grid, start=1):
    run_name = f"Phase2_{best_clf_name}_run{i:02d}"
    try:
        clf    = CLFClass(**clf_params)
        run_id, metrics = run_experiment(run_name, TFIDF_PARAMS, best_clf_name, clf)
        phase2_results.append({
            "run_id"  : run_id,
            "run_name": run_name,
            "clf_name": best_clf_name,
            **metrics
        })
    except Exception as e:
        print(f"  {run_name} gagal: {e}")


# PILIH MODEL TERBAIK & DAFTARKAN KE MODEL REGISTRY
all_results  = phase1_results + phase2_results
all_df       = pd.DataFrame(all_results).sort_values(BEST_METRIC, ascending=False)
best_overall = all_df.iloc[0]

print(f"\nBEST MODEL OVERALL")
print(f"  Run Name   : {best_overall['run_name']}")
print(f"  Classifier : {best_overall['clf_name']}")
print(f"  f1_macro   : {best_overall['f1_macro']:.4f}")
print(f"  Accuracy   : {best_overall['accuracy']:.4f}")

print("\nMendaftarkan model ke MLflow Model Registry ...")

model_uri  = f"runs:/{best_overall['run_id']}/{best_overall['run_name']}"
registered = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)

client = mlflow.tracking.MlflowClient()
client.update_model_version(
    name        = REGISTRY_NAME,
    version     = registered.version,
    description = (
        f"Best model — experiment: {EXPERIMENT_NAME}, "
        f"classifier: {best_overall['clf_name']}, "
        f"dataset: {DATASET_NAME} ({DATASET_VERSION}), "
        f"f1_macro: {best_overall['f1_macro']:.4f}, "
        f"accuracy: {best_overall['accuracy']:.4f}."
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

# RINGKASAN
print("\n" + "=" * 60)
print("RINGKASAN SEMUA EKSPERIMEN")
print("=" * 60)
summary_cols = ["run_name", "accuracy", "f1_macro", "f1_weighted", "cv_f1_macro_mean"]
available    = [c for c in summary_cols if c in all_df.columns]
print(all_df[available].to_string(index=False))

print("\nSelesai. Jalankan: mlflow ui  ->  http://localhost:5000")