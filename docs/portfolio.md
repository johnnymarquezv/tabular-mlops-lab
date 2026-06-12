# Portfolio Notes

Use this document to explain the project in a resume, interview, or portfolio
walkthrough.

## Short Description

`tabular-mlops-lab` is a local-first MLOps project that demonstrates the full
lifecycle of a tabular classification model: data preparation, training,
experiment tracking, artifact persistence, API inference, observability,
containerization, and local Kubernetes deployment.

## What It Demonstrates

- Built a reproducible ML pipeline with DVC.
- Trained and evaluated a scikit-learn tabular classifier.
- Tracked experiment parameters, metrics, and artifacts with MLflow.
- Persisted the serving model with `skops`.
- Exposed model inference through a FastAPI service.
- Added Prometheus-compatible request and latency metrics.
- Containerized the inference service with Docker.
- Deployed the service to a local `kind` Kubernetes cluster.
- Added CI-ready quality gates with Ruff, mypy, and pytest.

## Resume Bullets

- Built a local-first MLOps lab for tabular classification using scikit-learn,
  DVC, MLflow, FastAPI, Docker, and Kubernetes.
- Implemented reproducible data preparation and model training stages with DVC,
  producing versioned datasets, metrics, and model artifacts.
- Developed a FastAPI inference service with schema validation, health checks,
  Prometheus metrics, and Kubernetes deployment manifests.
- Added automated quality gates with Ruff, mypy, pytest, pre-commit, and GitHub
  Actions.

## Interview Talking Points

### Why DVC?

DVC makes the data and training pipeline reproducible. It records the
dependencies and outputs for each stage, so the project can determine whether
data preparation or training needs to run again.

### Why MLflow?

MLflow records training runs, metrics, parameters, and artifacts. It provides a
history of experiments instead of relying on terminal logs.

### Why FastAPI?

FastAPI gives a typed inference API, request validation with Pydantic, automatic
OpenAPI docs, and a simple local browser UI at `/docs`.

### Why skops?

The project uses `skops` for the serving artifact because it is a safer format
for scikit-learn model persistence than raw pickle or joblib.

### Why kind?

`kind` allows the same API container to be deployed to a local Kubernetes
cluster. This demonstrates deployment patterns without requiring a cloud account.

## Demo Script

1. Show the project layout and explain the separation between data, training,
   serving, metrics, and deployment.
2. Run data preparation:

```bash
.venv/bin/python -m mlops_tabular.data
```

3. Run training:

```bash
.venv/bin/python -m mlops_tabular.train
```

4. Open MLflow:

```bash
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

5. Start the API:

```bash
.venv/bin/uvicorn mlops_tabular.api:app --host 0.0.0.0 --port 8000 --reload
```

6. Open FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

7. Call `/predict` and explain the returned class, probabilities, and MLflow run
   id.

8. Show `/metrics` and explain the prediction count and latency metrics.

9. Optionally deploy to `kind`:

```bash
kind create cluster --name tabular-mlops-lab --config k8s/kind/cluster.yaml
docker build -t tabular-mlops-lab:latest .
kind load docker-image tabular-mlops-lab:latest --name tabular-mlops-lab
kubectl apply -k k8s/base
kubectl port-forward service/tabular-mlops-api 8000:80
```

## Possible Extensions

- Add a batch inference job.
- Add model drift checks.
- Add data validation with Great Expectations or Pandera.
- Add a feature store example.
- Deploy MLflow inside Kubernetes with PostgreSQL and object storage.
- Add a model promotion flow from staging to production.
- Add monitoring dashboards with Prometheus and Grafana.
- Add cloud deployment with Terraform.

## Disclaimer

The project uses a medical dataset for demonstration only. It is not intended for
clinical or diagnostic use.
