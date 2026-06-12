# Deployment

This project supports local API serving and local Kubernetes deployment with
Docker and `kind`.

## Local API Serving

The simplest serving mode runs FastAPI directly on your laptop:

```bash
.venv/bin/uvicorn mlops_tabular.api:app --host 0.0.0.0 --port 8000 --reload
```

Use this during development. It reads the trained model from:

```text
models/latest/model.skops
```

Endpoints:

- `GET /health`
- `POST /predict`
- `GET /metrics`
- `GET /docs`

## Docker Image

Build the API image:

```bash
docker build -t tabular-mlops-lab:latest .
```

The `Dockerfile`:

1. Starts from `python:3.11-slim`.
2. Copies the project into `/app`.
3. Installs the package with `python -m pip install --no-cache-dir .`.
4. Runs Uvicorn.

Container command:

```text
uvicorn mlops_tabular.api:app --host 0.0.0.0 --port 8000
```

Important note:

The image needs access to `models/latest/model.skops` at runtime. In this
scaffold, the generated model can be included in the image if it exists before
`docker build`. A more production-like setup would store model artifacts in an
artifact store and download or mount the selected model at deploy time.

## kind Cluster

Create the local Kubernetes cluster:

```bash
kind create cluster --name tabular-mlops-lab --config k8s/kind/cluster.yaml
```

`k8s/kind/cluster.yaml` defines:

- one control-plane node
- host port `8080` mapped to node port `30080`

This lets the NodePort service be reached from the host.

## Load The Image Into kind

`kind` does not automatically see images from the host Docker daemon. Load the
image into the cluster:

```bash
kind load docker-image tabular-mlops-lab:latest --name tabular-mlops-lab
```

## Deploy The API

Apply the Kubernetes manifests:

```bash
kubectl apply -k k8s/base
```

This creates:

- a `Deployment`
- a `Service`

Check the rollout:

```bash
kubectl get pods
kubectl get service tabular-mlops-api
```

Forward the service locally:

```bash
kubectl port-forward service/tabular-mlops-api 8000:80
```

Then use:

```text
http://127.0.0.1:8000
```

## Kubernetes Files

`k8s/base/deployment.yaml`

Runs the API container. It sets:

- image: `tabular-mlops-lab:latest`
- `MODEL_PATH=/app/models/latest/model.skops`
- liveness probe on `/health`
- readiness probe on `/health`
- CPU and memory requests/limits

`k8s/base/service.yaml`

Exposes the API as a `NodePort` service.

`k8s/base/kustomization.yaml`

Groups deployment and service resources so they can be applied together.

`k8s/kind/cluster.yaml`

Defines the local `kind` cluster and port mapping.

## Local MLflow Vs MLflow In kind

This project runs MLflow locally by default:

```bash
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Local MLflow means:

- runs are stored under `mlruns/`
- setup is simple
- it is good for learning and local development

Running MLflow inside `kind` would mean:

- MLflow runs as a Kubernetes deployment
- the UI is exposed through a service or port-forward
- the backend store needs persistence
- artifacts should live in durable storage

For a more production-like setup, MLflow would usually use:

- a database backend such as PostgreSQL
- object storage such as S3, GCS, Azure Blob, or MinIO
- Kubernetes manifests or Helm charts

That is intentionally not included in this scaffold to keep the project
local-first and easy to run.

## Cleanup

Delete API resources:

```bash
kubectl delete -k k8s/base --ignore-not-found
```

Delete the kind cluster:

```bash
kind delete cluster --name tabular-mlops-lab
```
