"""Model training and experiment tracking."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import skops.io as sio
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_tabular.config import (
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    REPORTS_DIR,
    Settings,
    settings,
)
from mlops_tabular.data import TARGET_COLUMN, TARGET_NAME_COLUMN, load_processed_data


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from a training run."""

    model_path: Path
    metrics_path: Path
    metrics: dict[str, float]
    run_id: str
    registered_model_name: str
    registered_model_version: str
    registered_model_alias: str


def _split_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_columns = [
        column for column in dataframe.columns if column not in {TARGET_COLUMN, TARGET_NAME_COLUMN}
    ]
    return dataframe[feature_columns], dataframe[TARGET_COLUMN].astype(int), feature_columns


def _target_names(dataframe: pd.DataFrame) -> list[str]:
    labels = (
        dataframe[[TARGET_COLUMN, TARGET_NAME_COLUMN]]
        .drop_duplicates()
        .sort_values(TARGET_COLUMN)[TARGET_NAME_COLUMN]
    )
    return [str(label) for label in labels.tolist()]


def _build_pipeline(active_settings: Settings) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=active_settings.n_estimators,
                    random_state=active_settings.random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model(active_settings: Settings = settings) -> TrainingResult:
    """Train a classifier, log the run to MLflow, and persist the latest model."""

    train_df, test_df = load_processed_data()
    x_train, y_train, feature_columns = _split_features_and_target(train_df)
    x_test, y_test, _ = _split_features_and_target(test_df)
    label_names = _target_names(train_df)

    pipeline = _build_pipeline(active_settings)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(active_settings.mlflow_tracking_uri)
    mlflow.set_experiment(active_settings.mlflow_experiment)

    with mlflow.start_run(run_name="random-forest-baseline") as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": active_settings.n_estimators,
                "random_state": active_settings.random_state,
                "test_size": active_settings.test_size,
                "feature_count": len(feature_columns),
            }
        )
        mlflow.log_metrics(metrics)

        signature = infer_signature(x_train, pipeline.predict(x_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            signature=signature,
            input_example=x_train.head(3),
        )
        model_version = mlflow.register_model(
            model_info.model_uri,
            active_settings.mlflow_registered_model_name,
            tags={"mlflow_run_id": run_id},
        )
        registered_model_version = str(model_version.version)

        model_bundle: dict[str, Any] = {
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "target_names": label_names,
            "metrics": metrics,
            "mlflow_run_id": run_id,
            "mlflow_registered_model_name": active_settings.mlflow_registered_model_name,
            "mlflow_registered_model_version": registered_model_version,
            "mlflow_model_alias": active_settings.mlflow_model_alias,
        }
        sio.dump(model_bundle, MODEL_PATH)

        METRICS_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")

    return TrainingResult(
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        metrics=metrics,
        run_id=run_id,
        registered_model_name=active_settings.mlflow_registered_model_name,
        registered_model_version=registered_model_version,
        registered_model_alias=active_settings.mlflow_model_alias,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tabular classification model.")
    parser.parse_args()
    result = train_model()
    print(f"MLflow run: {result.run_id}")
    print(
        "Registered model: "
        f"{result.registered_model_name} version {result.registered_model_version} "
        f"candidate for alias {result.registered_model_alias}"
    )
    print(f"Model saved to {result.model_path}")
    print(f"Metrics saved to {result.metrics_path}")
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
