from __future__ import annotations

from mlflow.tracking import MlflowClient

from mlops_tabular.config import settings
from mlops_tabular.data import prepare_data
from mlops_tabular.evaluate import evaluate_model
from mlops_tabular.promote_model import promote_model
from mlops_tabular.train import train_model
from mlops_tabular.validate import validate_data


def test_validate_data_writes_report() -> None:
    prepare_data()

    result = validate_data()

    assert result.valid is True
    assert result.raw_rows == result.train_rows + result.test_rows
    assert result.feature_count > 0


def test_evaluate_model_passes_thresholds() -> None:
    train_model()

    result = evaluate_model()

    assert result.passed is True
    assert result.metrics["accuracy"] >= settings.min_accuracy
    assert result.metrics["f1"] >= settings.min_f1
    assert result.metrics["roc_auc"] >= settings.min_roc_auc


def test_promote_model_sets_registry_alias() -> None:
    train_result = train_model()
    evaluate_model()

    result = promote_model()
    version = MlflowClient().get_model_version_by_alias(
        result.registered_model_name,
        result.registered_model_alias,
    )

    assert result.promoted is True
    assert result.registered_model_version == train_result.registered_model_version
    assert str(version.version) == train_result.registered_model_version
