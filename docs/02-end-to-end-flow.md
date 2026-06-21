# End-To-End Flow

This document follows one model run through the whole project.

## 1. Environment Setup

Create an environment outside the project and install the package:

```bash
python3 -m venv ~/.venvs/tabular-mlops-lab
source ~/.venvs/tabular-mlops-lab/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

After this, Python can import `mlops_tabular` from `src/mlops_tabular`.

## 2. Data Preparation

Run:

```bash
python3 -m mlops_tabular.data
```

Python executes `src/mlops_tabular/data.py`.

The module:

1. Loads scikit-learn's breast cancer dataset.
2. Converts feature names to snake case.
3. Adds `target` and `target_name`.
4. Splits the data into train and test sets.
5. Writes CSV files under `data/`.

## 3. Training

Run:

```bash
python3 -m mlops_tabular.train
```

Python executes `src/mlops_tabular/train.py`.

The module:

1. Loads `data/processed/train.csv` and `data/processed/test.csv`.
2. Builds a scikit-learn pipeline.
3. Fits the pipeline on training data.
4. Predicts against test data.
5. Calculates metrics.
6. Starts an MLflow run.
7. Logs parameters and metrics.
8. Logs the model artifact to MLflow.
9. Registers a new `tabular-mlops-classifier` model version.
10. Moves the `champion` alias to that version.
11. Saves `models/latest/model.skops`.
12. Saves `reports/metrics.json`.

## 4. Experiment Tracking

MLflow writes run metadata under `mlruns/`.

You can inspect it with:

```bash
PYTHONPATH=. MLFLOW_ALLOW_FILE_STORE=true python3 -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

MLflow answers questions such as:

- Which parameters did this run use?
- What metrics did this run produce?
- Which run produced the model currently being served?
- Which registered model version and alias point at this model?

## 5. Model Serving

Run:

```bash
python3 -m uvicorn mlops_tabular.api:app --host 0.0.0.0 --port 8000 --reload
```

FastAPI imports `src/mlops_tabular/api.py`.

When a request reaches `POST /predict`:

1. `api.py` receives and validates the JSON request with Pydantic.
2. `api.py` calls `predict()` from `model.py`.
3. `model.py` loads `models/latest/model.skops` if it is not already cached.
4. `model.py` checks that all expected features are present.
5. `model.py` creates a pandas dataframe.
6. The scikit-learn pipeline predicts the class and probabilities.
7. `api.py` returns the response as JSON.
8. `metrics.py` records request count and latency.

## 6. Pipeline Reproduction

Run:

```bash
python3 -m dvc repro
```

DVC reads `dvc.yaml` and runs stages only when needed.

If `data.py` changes, DVC reruns `prepare_data` and then `train`.

If only `train.py` changes, DVC reruns `train`.

If nothing changed, DVC reuses cached outputs.

## 7. Container And Kubernetes

Build the image:

```bash
docker build -t tabular-mlops-lab:latest .
```

Select the OrbStack Kubernetes context and deploy:

```bash
kubectl config use-context orbstack
kubectl apply -k k8s/base
kubectl port-forward service/tabular-mlops-api 8000:80
```

At that point, the same API runs as a Kubernetes workload.

## Summary

The core flow is:

```text
src/mlops_tabular/data.py
  -> data/
  -> src/mlops_tabular/train.py
  -> models/latest/model.skops
  -> reports/metrics.json
  -> mlruns/
  -> src/mlops_tabular/api.py
  -> /predict
```
