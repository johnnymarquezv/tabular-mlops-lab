from __future__ import annotations

from mlops_tabular.train import train_model


def test_train_model_persists_artifacts_and_metrics() -> None:
    result = train_model()

    assert result.model_path.exists()
    assert result.metrics_path.exists()
    assert result.run_id
    assert result.registered_model_name == "tabular-mlops-classifier"
    assert result.registered_model_version
    assert result.registered_model_alias == "champion"
    assert result.metrics["accuracy"] >= 0.9
    assert result.metrics["f1"] >= 0.9
    assert result.metrics["roc_auc"] >= 0.9
