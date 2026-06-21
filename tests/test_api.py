from __future__ import annotations

from fastapi.testclient import TestClient

from mlops_tabular.api import app
from mlops_tabular.data import TARGET_COLUMN, TARGET_NAME_COLUMN, load_processed_data
from mlops_tabular.model import clear_model_cache
from mlops_tabular.train import train_model


def _sample_features() -> dict[str, float]:
    _, test_df = load_processed_data()
    row = test_df.drop(columns=[TARGET_COLUMN, TARGET_NAME_COLUMN]).iloc[0]
    return {str(name): float(value) for name, value in row.items()}


def test_health_reports_loaded_model() -> None:
    train_model()
    clear_model_cache()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["mlflow_run_id"]
    assert body["registered_model_name"] == "tabular-mlops-classifier"
    assert body["registered_model_version"]
    assert body["registered_model_alias"] == "champion"


def test_predict_returns_class_and_probabilities() -> None:
    train_model()
    clear_model_cache()
    client = TestClient(app)

    response = client.post("/predict", json={"features": _sample_features()})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in {0, 1}
    assert body["predicted_label"] in {"malignant", "benign"}
    assert set(body["probabilities"]) == {"malignant", "benign"}
    assert body["registered_model_name"] == "tabular-mlops-classifier"
    assert body["registered_model_version"]
    assert body["registered_model_alias"] == "champion"
