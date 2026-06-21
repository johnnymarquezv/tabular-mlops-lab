"""Project configuration and filesystem paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "latest"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "model.skops"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by data preparation, training, and serving."""

    random_state: int = _env_int("MLOPS_RANDOM_STATE", 42)
    test_size: float = _env_float("MLOPS_TEST_SIZE", 0.2)
    n_estimators: int = _env_int("MLOPS_N_ESTIMATORS", 200)
    mlflow_experiment: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "tabular-classification")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow_registered_model_name: str = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "tabular-mlops-classifier",
    )
    mlflow_model_alias: str = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
    model_path: Path = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))


settings = Settings()
