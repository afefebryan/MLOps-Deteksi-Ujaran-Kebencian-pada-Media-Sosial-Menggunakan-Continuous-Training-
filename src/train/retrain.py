import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/tdavidson_hate_speech_v0_clean.csv"
EXPERIMENT_NAME = "hate-speech-classification"
BEST_METRIC = "f1_macro"

TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

# =========================
# MLflow SAFE MODE (CI)
# =========================
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MLRUNS_PATH = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
X = df["clean_tweet"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }

# =========================
# PIPELINE
# =========================
def run_model(name, model):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf", model)
    ])

    with mlflow.start_run(run_name=name):
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        metrics = compute_metrics(y_test, preds)

        mlflow.log_metrics(metrics)
        mlflow.log_param("model", name)

        print(f"{name} -> f1_macro={metrics['f1_macro']:.4f}")

        return metrics

# =========================
# MODELS
# =========================
models = {
    "LR": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "LinearSVC": LinearSVC(),
    "SGD": SGDClassifier(loss="hinge"),
    "RF": RandomForestClassifier(n_estimators=100),
}

# =========================
# TRAIN ALL
# =========================
results = []

print("\n=== CI TRAINING START ===")

for name, model in models.items():
    metrics = run_model(name, model)
    results.append((name, metrics["f1_macro"]))

best = sorted(results, key=lambda x: x[1], reverse=True)[0]

print("\n=== BEST MODEL ===")
print(best)

# =========================
# OUTPUT FOR GITHUB ACTIONS
# =========================
with open("best_model.txt", "w") as f:
    f.write(str(best))