"""FastAPI inference service."""

from __future__ import annotations

import logging
import os
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pythonjsonlogger.json import JsonFormatter
from starlette.responses import Response

from mlops_tabular.config import MODEL_PATH
from mlops_tabular.metrics import prometheus_response, record_prediction
from mlops_tabular.model import get_model_bundle, predict


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


_configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Tabular Inference API",
    version="0.1.0",
    description="Online inference API for the demo tabular classification model.",
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    mlflow_run_id: str | None = None
    metrics: dict[str, float] | None = None


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        examples=[
            {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189,
            }
        ]
    )


class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    mlflow_run_id: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        bundle = get_model_bundle()
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False, model_path=str(MODEL_PATH))

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_path=str(MODEL_PATH),
        mlflow_run_id=bundle.mlflow_run_id,
        metrics=bundle.metrics,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    started_at = time.perf_counter()
    try:
        prediction = predict(request.features)
    except FileNotFoundError as error:
        record_prediction("model_missing", time.perf_counter() - started_at)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        record_prediction("invalid_request", time.perf_counter() - started_at)
        raise HTTPException(status_code=422, detail=str(error)) from error

    elapsed_seconds = time.perf_counter() - started_at
    record_prediction("success", elapsed_seconds)
    logger.info(
        "prediction_completed",
        extra={
            "predicted_class": prediction.predicted_class,
            "predicted_label": prediction.predicted_label,
            "latency_seconds": elapsed_seconds,
        },
    )
    return PredictResponse(**prediction.__dict__)


@app.get("/metrics", response_class=Response)
def metrics() -> Response:
    return prometheus_response()


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("mlops_tabular.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
