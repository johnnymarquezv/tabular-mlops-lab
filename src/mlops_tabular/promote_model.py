"""Promote an evaluated model version in the MLflow Model Registry."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from mlops_tabular.config import EVALUATION_PATH, PROMOTION_PATH, REPORTS_DIR, Settings, settings
from mlops_tabular.model import load_model_bundle


@dataclass(frozen=True)
class PromotionResult:
    """Registered model promotion result."""

    report_path: str
    registered_model_name: str
    registered_model_version: str
    registered_model_alias: str
    promoted: bool


def _load_evaluation_report() -> dict[str, float]:
    raw_report = json.loads(EVALUATION_PATH.read_text())
    return {str(key): float(value) for key, value in raw_report.items()}


def promote_model(active_settings: Settings = settings) -> PromotionResult:
    """Move the configured model alias after evaluation has passed."""

    evaluation = _load_evaluation_report()
    if int(evaluation.get("passed", 0)) != 1:
        raise ValueError("Cannot promote model because evaluation did not pass")

    bundle = load_model_bundle()
    if bundle.registered_model_name is None or bundle.registered_model_version is None:
        raise ValueError("Model bundle does not include registered model metadata")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(active_settings.mlflow_tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(
        bundle.registered_model_name,
        active_settings.mlflow_model_alias,
        bundle.registered_model_version,
    )

    report = {
        "promoted": 1,
        "registered_model_name": bundle.registered_model_name,
        "registered_model_version": bundle.registered_model_version,
        "registered_model_alias": active_settings.mlflow_model_alias,
        "accuracy": evaluation["accuracy"],
        "f1": evaluation["f1"],
        "roc_auc": evaluation["roc_auc"],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROMOTION_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return PromotionResult(
        report_path=str(PROMOTION_PATH),
        registered_model_name=bundle.registered_model_name,
        registered_model_version=bundle.registered_model_version,
        registered_model_alias=active_settings.mlflow_model_alias,
        promoted=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the evaluated model registry version.")
    parser.parse_args()
    result = promote_model()
    print(
        "Promoted registered model "
        f"{result.registered_model_name} version {result.registered_model_version} "
        f"as alias {result.registered_model_alias}"
    )
    print(f"Promotion report saved to {result.report_path}")


if __name__ == "__main__":
    main()
