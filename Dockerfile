FROM python:3-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY . .

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "mlops_tabular.api:app", "--host", "0.0.0.0", "--port", "8000"]
