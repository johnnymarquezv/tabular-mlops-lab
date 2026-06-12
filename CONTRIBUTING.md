# Contributing

Thanks for taking a look at `tabular-mlops-lab`.

This is primarily a portfolio and learning project, but issues and suggestions
are welcome.

## Local Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

## Checks

Run these before opening a pull request:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src tests
.venv/bin/python -m pytest
```

## Development Notes

- Keep source code under `src/mlops_tabular/`.
- Keep generated data, models, reports, and MLflow runs out of Git.
- Update docs when changing workflows, artifacts, or deployment commands.
- This project is educational and should not be used for medical decisions.
