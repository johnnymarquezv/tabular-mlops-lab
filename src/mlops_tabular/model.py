"""Model loading and prediction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import skops.io as sio

from mlops_tabular.config import MODEL_PATH


@dataclass(frozen=True)
class ModelBundle:
    """Serializable model metadata needed for inference."""

    pipeline: Any
    feature_columns: list[str]
    target_names: list[str]
    metrics: dict[str, float]
    mlflow_run_id: str
    registered_model_name: str | None = None
    registered_model_version: str | None = None
    registered_model_alias: str | None = None


@dataclass(frozen=True)
class PredictionResult:
    """Normalized model prediction output."""

    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    mlflow_run_id: str
    registered_model_name: str | None = None
    registered_model_version: str | None = None
    registered_model_alias: str | None = None


def load_model_bundle(model_path: Path = MODEL_PATH) -> ModelBundle:
    """Load a trained model bundle from disk."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run `python3 -m mlops_tabular.train` first."
        )

    trusted_types = sio.get_untrusted_types(file=model_path)
    raw_bundle = sio.load(model_path, trusted=trusted_types)
    if not isinstance(raw_bundle, dict):
        raise TypeError(f"Expected model bundle dictionary, got {type(raw_bundle)!r}.")

    return ModelBundle(
        pipeline=raw_bundle["pipeline"],
        feature_columns=[str(column) for column in raw_bundle["feature_columns"]],
        target_names=[str(name) for name in raw_bundle["target_names"]],
        metrics={str(name): float(value) for name, value in raw_bundle["metrics"].items()},
        mlflow_run_id=str(raw_bundle["mlflow_run_id"]),
        registered_model_name=(
            str(raw_bundle["mlflow_registered_model_name"])
            if "mlflow_registered_model_name" in raw_bundle
            else None
        ),
        registered_model_version=(
            str(raw_bundle["mlflow_registered_model_version"])
            if "mlflow_registered_model_version" in raw_bundle
            else None
        ),
        registered_model_alias=(
            str(raw_bundle["mlflow_model_alias"]) if "mlflow_model_alias" in raw_bundle else None
        ),
    )


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    """Cache the model bundle for API serving."""

    return load_model_bundle()


def clear_model_cache() -> None:
    """Clear the model cache, mainly for tests."""

    get_model_bundle.cache_clear()


def predict(features: Mapping[str, float], bundle: ModelBundle | None = None) -> PredictionResult:
    """Run inference after validating the request feature schema."""

    active_bundle = bundle or get_model_bundle()
    provided_features = set(features)
    expected_features = set(active_bundle.feature_columns)
    missing_features = sorted(expected_features - provided_features)
    extra_features = sorted(provided_features - expected_features)

    if missing_features or extra_features:
        details = []
        if missing_features:
            details.append(f"missing features: {', '.join(missing_features)}")
        if extra_features:
            details.append(f"unexpected features: {', '.join(extra_features)}")
        raise ValueError("; ".join(details))

    frame = pd.DataFrame(
        [{column: float(features[column]) for column in active_bundle.feature_columns}]
    )
    predicted_class = int(active_bundle.pipeline.predict(frame)[0])
    probability_values = active_bundle.pipeline.predict_proba(frame)[0]
    probabilities = {
        active_bundle.target_names[index]: float(probability)
        for index, probability in enumerate(probability_values)
    }
    predicted_label = active_bundle.target_names[predicted_class]

    return PredictionResult(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        probabilities=probabilities,
        mlflow_run_id=active_bundle.mlflow_run_id,
        registered_model_name=active_bundle.registered_model_name,
        registered_model_version=active_bundle.registered_model_version,
        registered_model_alias=active_bundle.registered_model_alias,
    )
