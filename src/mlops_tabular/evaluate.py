"""Model evaluation gate for the DVC pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from mlops_tabular.config import EVALUATION_PATH, REPORTS_DIR, Settings, settings
from mlops_tabular.data import TARGET_COLUMN, TARGET_NAME_COLUMN, load_processed_data
from mlops_tabular.model import load_model_bundle


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics and gate result."""

    report_path: str
    metrics: dict[str, float]
    passed: bool


def _passes_thresholds(metrics: dict[str, float], active_settings: Settings) -> bool:
    return (
        metrics["accuracy"] >= active_settings.min_accuracy
        and metrics["f1"] >= active_settings.min_f1
        and metrics["roc_auc"] >= active_settings.min_roc_auc
    )


def evaluate_model(active_settings: Settings = settings) -> EvaluationResult:
    """Evaluate the saved serving model and enforce metric thresholds."""

    _, test_df = load_processed_data()
    bundle = load_model_bundle()
    x_test = test_df.drop(columns=[TARGET_COLUMN, TARGET_NAME_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    predictions = bundle.pipeline.predict(x_test)
    probabilities = bundle.pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    passed = _passes_thresholds(metrics, active_settings)
    report = {
        **metrics,
        "passed": int(passed),
        "min_accuracy": active_settings.min_accuracy,
        "min_f1": active_settings.min_f1,
        "min_roc_auc": active_settings.min_roc_auc,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if not passed:
        raise ValueError(f"Model failed evaluation thresholds: {json.dumps(report)}")

    return EvaluationResult(report_path=str(EVALUATION_PATH), metrics=metrics, passed=passed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the latest trained model.")
    parser.parse_args()
    result = evaluate_model()
    print(f"Evaluation report saved to {result.report_path}")
    print(json.dumps({**result.metrics, "passed": int(result.passed)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
