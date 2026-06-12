# Workflows

This file explains how to run the project with direct virtual environment
commands.

## 1. Create The Environment

From the project root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

This creates `.venv/`, installs the package in editable mode, and installs
development tools such as pytest, Ruff, mypy, DVC, and MLflow.

Editable mode matters because the package source lives under `src/`. After
installation, Python can import `mlops_tabular` from that source directory.

## 2. Prepare Data

```bash
.venv/bin/python -m mlops_tabular.data
```

This runs `src/mlops_tabular/data.py`.

Outputs:

- `data/raw/breast_cancer.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`

The command loads scikit-learn's built-in breast cancer dataset, cleans the
feature names, adds target labels, and creates stratified train/test splits.

## 3. Train The Model

```bash
.venv/bin/python -m mlops_tabular.train
```

This runs `src/mlops_tabular/train.py`.

Outputs:

- `models/latest/model.skops`
- `reports/metrics.json`
- `mlruns/`

Training builds a scikit-learn pipeline with `StandardScaler` and
`RandomForestClassifier`. It evaluates the model on the test split and logs the
run to MLflow.

## 4. Serve The Model Locally

```bash
.venv/bin/uvicorn mlops_tabular.api:app --host 0.0.0.0 --port 8000 --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Useful endpoints:

- `GET /health`
- `POST /predict`
- `GET /metrics`
- `GET /docs`

`/docs` opens FastAPI's browser UI where you can call `POST /predict` without
writing a curl command.

## 5. Use The Model Through The API

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"mean_radius":17.99,"mean_texture":10.38,"mean_perimeter":122.8,"mean_area":1001.0,"mean_smoothness":0.1184,"mean_compactness":0.2776,"mean_concavity":0.3001,"mean_concave_points":0.1471,"mean_symmetry":0.2419,"mean_fractal_dimension":0.07871,"radius_error":1.095,"texture_error":0.9053,"perimeter_error":8.589,"area_error":153.4,"smoothness_error":0.006399,"compactness_error":0.04904,"concavity_error":0.05373,"concave_points_error":0.01587,"symmetry_error":0.03003,"fractal_dimension_error":0.006193,"worst_radius":25.38,"worst_texture":17.33,"worst_perimeter":184.6,"worst_area":2019.0,"worst_smoothness":0.1622,"worst_compactness":0.6656,"worst_concavity":0.7119,"worst_concave_points":0.2654,"worst_symmetry":0.4601,"worst_fractal_dimension":0.1189}}'
```

Example response:

```json
{
  "predicted_class": 0,
  "predicted_label": "malignant",
  "probabilities": {
    "malignant": 0.98,
    "benign": 0.02
  },
  "mlflow_run_id": "..."
}
```

## 6. View MLflow

```bash
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Use MLflow to inspect runs, parameters, metrics, and logged artifacts. MLflow is
not the inference UI; the inference UI is FastAPI's `/docs` page.

## 7. Reproduce The Pipeline With DVC

```bash
.venv/bin/dvc repro
```

DVC reads `dvc.yaml`, checks whether dependencies changed, and runs only the
stages that need to be refreshed.

The two stages are:

- `prepare_data`
- `train`

DVC updates `dvc.lock` after a successful run.

## 8. Run Quality Checks

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src tests
.venv/bin/python -m pytest
```

These checks are also represented in GitHub Actions.

## 9. Useful Console Scripts

The project defines these console scripts in `pyproject.toml`:

- `.venv/bin/mlops-prepare-data`
- `.venv/bin/mlops-train`
- `.venv/bin/mlops-serve`

The documentation uses `python -m ...` commands because they show clearly which
module runs. The console scripts are equivalent shortcuts.
