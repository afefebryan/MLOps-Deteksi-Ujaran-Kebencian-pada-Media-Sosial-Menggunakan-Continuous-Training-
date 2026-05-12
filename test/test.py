"""
tests/test_train.py
Unit tests untuk memverifikasi integritas pipeline sebelum training.
Jalankan: pytest tests/test_train.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

DATA_PATH = "data/processed/tdavidson_hate_speech_v0_clean.csv"

# ---------------------------------------------------------------
# FIXTURE
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset():
    df = pd.read_csv(DATA_PATH)
    return df

@pytest.fixture(scope="module")
def split_data(dataset):
    X = dataset["clean_tweet"]
    y = dataset["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

@pytest.fixture(scope="module")
def trained_pipeline(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


# ---------------------------------------------------------------
# TEST 1 — Dataset
# ---------------------------------------------------------------
class TestDataset:
    def test_file_exists(self):
        assert os.path.exists(DATA_PATH), f"File tidak ditemukan: {DATA_PATH}"

    def test_required_columns(self, dataset):
        assert "clean_tweet" in dataset.columns, "Kolom 'clean_tweet' tidak ada"
        assert "label" in dataset.columns, "Kolom 'label' tidak ada"

    def test_no_null_values(self, dataset):
        assert dataset["clean_tweet"].isnull().sum() == 0, "Ada nilai null di clean_tweet"
        assert dataset["label"].isnull().sum() == 0, "Ada nilai null di label"

    def test_label_is_binary(self, dataset):
        unique_labels = set(dataset["label"].unique())
        assert unique_labels == {0, 1}, f"Label bukan binary: {unique_labels}"

    def test_minimum_rows(self, dataset):
        assert len(dataset) >= 100, f"Dataset terlalu kecil: {len(dataset)} baris"

    def test_tweet_is_string(self, dataset):
        assert dataset["clean_tweet"].dtype == object, "clean_tweet bukan tipe string"


# ---------------------------------------------------------------
# TEST 2 — Data Split
# ---------------------------------------------------------------
class TestDataSplit:
    def test_split_ratio(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        total = len(X_train) + len(X_test)
        ratio = len(X_test) / total
        assert abs(ratio - 0.3) < 0.01, f"Rasio split tidak sesuai: {ratio:.2f}"

    def test_no_data_leakage(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, "Ada data leakage antara train dan test"

    def test_stratification(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        train_ratio = y_train.mean()
        test_ratio  = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05, "Stratifikasi label tidak seimbang"


# ---------------------------------------------------------------
# TEST 3 — Pipeline
# ---------------------------------------------------------------
class TestPipeline:
    def test_pipeline_fit(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        assert pipeline is not None

    def test_pipeline_predict_shape(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        preds = pipeline.predict(X_test)
        assert len(preds) == len(X_test), "Jumlah prediksi tidak sama dengan jumlah input"

    def test_pipeline_predict_binary(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        preds = pipeline.predict(X_test)
        unique_preds = set(np.unique(preds))
        assert unique_preds.issubset({0, 1}), f"Output prediksi bukan binary: {unique_preds}"

    def test_pipeline_no_nan_output(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        preds = pipeline.predict(X_test)
        assert not np.any(np.isnan(preds.astype(float))), "Ada NaN di output prediksi"


# ---------------------------------------------------------------
# TEST 4 — Metrik
# ---------------------------------------------------------------
class TestMetrics:
    def test_f1_macro_above_threshold(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        preds = pipeline.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")
        assert f1 >= 0.85, f"f1_macro {f1:.4f} di bawah threshold 0.85"

    def test_accuracy_above_threshold(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        assert acc >= 0.90, f"Accuracy {acc:.4f} di bawah threshold 0.90"

    def test_both_classes_predicted(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        preds = pipeline.predict(X_test)
        assert 0 in preds and 1 in preds, "Model hanya memprediksi satu kelas"