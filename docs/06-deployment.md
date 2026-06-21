# Deployment

This project supports local API serving and local Kubernetes deployment with
Docker and OrbStack.

## Local API Serving

The simplest serving mode runs FastAPI directly on your laptop:

```bash
python3 -m uvicorn mlops_tabular.api:app --host 0.0.0.0 --port 8000 --reload
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

1. Starts from `python:3-slim`.
2. Copies the project into `/app`.
3. Installs the package with `python3 -m pip install --no-cache-dir .`.
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

## OrbStack Kubernetes

Enable Kubernetes in OrbStack, then point `kubectl` at the OrbStack context:

```bash
kubectl config use-context orbstack
kubectl config current-context
```

Build the API image after the model artifact exists:

```bash
docker build -t tabular-mlops-lab:latest .
```

OrbStack's Kubernetes cluster can use images from the local OrbStack Docker
engine, so there is no separate image load step.

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

The service is also defined as a `NodePort` on port `30080`, which can be useful
when you want to test without a port-forward.

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

The manifests under `k8s/base` are not tied to a specific local Kubernetes
provider. OrbStack supplies the cluster and Docker image runtime for local
development.

## Local MLflow Vs MLflow In Kubernetes

This project runs MLflow locally by default:

```bash
PYTHONPATH=. MLFLOW_ALLOW_FILE_STORE=true python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Local MLflow means:

- runs are stored under `mlruns/`
- setup is simple
- it is good for learning and local development

Running MLflow inside Kubernetes would mean:

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

To stop the local cluster itself, disable Kubernetes in OrbStack.
