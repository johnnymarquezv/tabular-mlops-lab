"""Prometheus metrics for the inference service."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

PREDICTION_COUNTER = Counter(
    "mlops_tabular_predictions_total",
    "Total prediction requests by outcome.",
    labelnames=("status",),
)
PREDICTION_LATENCY = Histogram(
    "mlops_tabular_prediction_latency_seconds",
    "Prediction request latency in seconds.",
)


def record_prediction(status: str, elapsed_seconds: float) -> None:
    """Record the result of a prediction request."""

    PREDICTION_COUNTER.labels(status=status).inc()
    PREDICTION_LATENCY.observe(elapsed_seconds)


def prometheus_response() -> Response:
    """Render all registered Prometheus metrics."""

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
