# Artifacts And Outputs

This project creates several generated directories when you run data preparation,
training, MLflow, DVC, or the API. These outputs are intentionally separated
from source code.

## `data/`

Created by:

```bash
.venv/bin/python -m mlops_tabular.data
```

Also created or refreshed by:

```bash
.venv/bin/dvc repro
```

Purpose:

`data/` stores the dataset used by training.

Files:

- `data/raw/breast_cancer.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`

`data/raw/breast_cancer.csv`

The full cleaned dataset from scikit-learn. It includes normalized feature names,
the numeric `target`, and the human-readable `target_name`.

`data/processed/train.csv`

The training split. The model fits on this file.

`data/processed/test.csv`

The test split. Training uses this file to calculate evaluation metrics.

Why it matters:

Training data is an input to the model. If the data changes, model behavior can
change even when code stays the same.

## `models/`

Created by:

```bash
.venv/bin/python -m mlops_tabular.train
```

Purpose:

`models/` stores the latest model artifact used by the API.

Files:

- `models/latest/model.skops`

`model.skops` contains:

- the trained scikit-learn pipeline
- ordered feature names
- target names
- latest metrics
- MLflow run id

The API reads this file when serving predictions. If it is missing, `/health`
returns a degraded state and `/predict` cannot run inference.

Why `skops`:

`skops` is a safer serialization format for scikit-learn models than raw pickle
or joblib. It still requires trust decisions when loading, but it avoids using
plain pickle as the project-level model artifact.

## `reports/`

Created by:

```bash
.venv/bin/python -m mlops_tabular.train
```

Purpose:

`reports/` stores readable training outputs.

Files:

- `reports/metrics.json`

Example:

```json
{
  "accuracy": 0.9473684210526315,
  "f1": 0.9577464788732394,
  "roc_auc": 0.9943783068783069
}
```

DVC treats this file as a metric. That means model quality can be tracked as a
pipeline output.

## `mlruns/`

Created by:

```bash
.venv/bin/python -m mlops_tabular.train
```

Purpose:

`mlruns/` is MLflow's local file-backed tracking store.

It contains:

- experiment ids
- run ids
- parameters
- metrics
- model metadata
- logged artifacts

Open it with:

```bash
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Then visit:

```text
http://127.0.0.1:5000
```

Do not manually edit files inside `mlruns/`. Use the MLflow UI or MLflow APIs.

## `.venv/`

Created during setup:

```bash
python3 -m venv .venv
```

Purpose:

`.venv/` is the local Python environment. It contains installed dependencies and
command-line tools such as:

- `.venv/bin/python`
- `.venv/bin/pytest`
- `.venv/bin/ruff`
- `.venv/bin/mypy`
- `.venv/bin/dvc`
- `.venv/bin/mlflow`
- `.venv/bin/uvicorn`

This directory is local-only and ignored by Git.

## `.dvc/`

Created by DVC initialization.

Purpose:

`.dvc/` stores DVC repository metadata and configuration. It is source-controlled
except for the DVC cache.

The DVC cache stores content-addressed versions of outputs. It should not be
edited manually.

## `dvc.lock`

Created or updated by:

```bash
.venv/bin/dvc repro
```

Purpose:

`dvc.lock` records the latest known-good pipeline state.

It stores:

- commands that ran
- dependencies for each stage
- output files for each stage
- hashes and sizes

If a dependency hash changes, DVC knows the stage may need to run again.

## API Runtime Outputs

When the API runs, it does not write new model files by default. It reads
`models/latest/model.skops` and emits:

- JSON logs to stdout
- Prometheus metrics from `/metrics`
- HTTP responses from `/health` and `/predict`

The API is stateless with respect to training. To change the model, run training
again and restart the API process if needed.

## What Should Be Committed

Normally commit:

- source code
- tests
- `pyproject.toml`
- `dvc.yaml`
- `dvc.lock`
- Docker and Kubernetes manifests
- documentation

Normally do not commit:

- `.venv/`
- `data/`
- `models/`
- `reports/`
- `mlruns/`
- Python caches

Those generated outputs are ignored by `.gitignore`.
