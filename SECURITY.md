# Security

This is an educational MLOps portfolio project.

## Supported Use

The project demonstrates local model training and API inference. It is not
intended for production medical, clinical, or diagnostic use.

## Reporting Issues

If you find a security issue, please open a GitHub issue with enough detail to
reproduce it, but avoid posting secrets, credentials, or sensitive data.

## Model Artifact Safety

The project uses `skops` for scikit-learn model persistence instead of raw pickle
or joblib. Still, only load model artifacts that you trust.

## Secrets

Do not commit secrets. Local environment files such as `.env` are ignored by
Git. Use `.env.example` as a template for non-sensitive configuration names.
