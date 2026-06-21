"""Compatibility shims loaded automatically by Python when run from this repo."""

from __future__ import annotations

import importlib
import importlib.abc

try:
    resources_abc = importlib.import_module("importlib.resources.abc")
    traversable = getattr(resources_abc, "Traversable", None)
except ImportError:  # pragma: no cover - kept for older Python compatibility.
    traversable = None

if traversable is not None and not hasattr(importlib.abc, "Traversable"):
    # Some MLflow versions still import Traversable from the old Python location.
    importlib.abc.Traversable = traversable  # type: ignore[attr-defined]
