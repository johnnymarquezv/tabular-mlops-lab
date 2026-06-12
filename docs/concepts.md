# Concepts

This project is a compact MLOps system. It is small enough to run locally, but
the workflow mirrors the shape of larger production ML systems.

## What The Project Does

The model predicts whether a tumor sample is `malignant` or `benign` using the
scikit-learn breast cancer dataset.

The end-to-end flow is:

```text
data preparation -> training -> experiment tracking -> model artifact -> API serving -> deployment
```

Each stage has a separate responsibility:

- Data preparation creates stable train/test CSV files.
- Training builds and evaluates the model.
- MLflow records parameters, metrics, and model metadata.
- The model artifact stores the trained pipeline used by inference.
- FastAPI exposes the model through HTTP.
- Prometheus metrics expose API usage and latency.
- Docker and Kubernetes package and run the service outside the Python process.

## MLOps Concepts Used

### Reproducible Data

The project uses DVC to define the data and training pipeline. DVC tracks which
files and commands belong to each stage. If code or data changes, `dvc repro`
knows which stages need to run again.

This matters because ML results depend on both code and data. Tracking only code
is not enough.

### Experiment Tracking

MLflow records each training run. A run contains:

- parameters such as model type and number of trees
- metrics such as accuracy, F1, and ROC AUC
- artifacts such as evaluation reports and model metadata

This lets you compare model runs instead of relying on terminal output.

### Model Artifact

Training saves the latest model to `models/latest/model.skops`. The API loads
that file at runtime.

The saved artifact includes:

- the trained scikit-learn pipeline
- feature names expected by the model
- target labels
- evaluation metrics
- the MLflow run id that produced the model

The project uses `skops` instead of pickle/joblib for model persistence because
it is safer for scikit-learn model serialization.

### Online Inference

The FastAPI service exposes the model through HTTP. A client sends feature
values to `POST /predict`; the service validates the feature schema, runs the
model, and returns the prediction.

This is online inference: predictions happen request-by-request.

### Observability

The service exposes `/metrics` in Prometheus format. It records:

- prediction request counts by status
- prediction latency

This is the application side of ML observability. In a larger system, you would
also monitor model quality, input drift, output drift, and data quality.

### Local Kubernetes

`kind` runs a Kubernetes cluster locally. The project uses it to show how the
API can be packaged as a container and deployed as a Kubernetes workload.

MLflow stays local in this scaffold. Running MLflow inside Kubernetes is possible
but requires extra persistence and backend services.

## Tool Choices

- Python `venv` and `pip`: native Python environment management.
- scikit-learn: simple, widely used ML library for tabular models.
- DVC: reproducible data and model pipeline stages.
- MLflow: experiment tracking and model metadata.
- skops: safer scikit-learn model serialization.
- FastAPI: typed HTTP API for inference.
- Prometheus client: metrics endpoint for service monitoring.
- Docker: container image for the API.
- kind and kubectl: local Kubernetes deployment.
- Ruff, mypy, pytest: linting, type checking, and tests.
