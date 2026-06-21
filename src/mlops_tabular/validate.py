"""Data validation checks for the DVC pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from mlops_tabular.config import DATA_VALIDATION_PATH, REPORTS_DIR
from mlops_tabular.data import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TARGET_NAME_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    load_demo_dataframe,
)


@dataclass(frozen=True)
class ValidationResult:
    """Summary of data validation checks."""

    report_path: str
    raw_rows: int
    train_rows: int
    test_rows: int
    feature_count: int
    valid: bool


def _validate_frame(name: str, dataframe: pd.DataFrame, expected_columns: list[str]) -> None:
    missing_columns = sorted(set(expected_columns) - set(dataframe.columns))
    extra_columns = sorted(set(dataframe.columns) - set(expected_columns))
    if missing_columns or extra_columns:
        raise ValueError(
            f"{name} schema mismatch: missing={missing_columns}, extra={extra_columns}"
        )

    if dataframe.empty:
        raise ValueError(f"{name} is empty")

    if dataframe.isna().any().any():
        raise ValueError(f"{name} contains null values")

    if set(dataframe[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError(f"{name} must contain both target classes")

    expected_target_names = {"malignant", "benign"}
    if set(dataframe[TARGET_NAME_COLUMN].unique()) != expected_target_names:
        raise ValueError(f"{name} has unexpected target labels")


def validate_data() -> ValidationResult:
    """Validate raw and processed datasets produced by data preparation."""

    expected_columns = [str(column) for column in load_demo_dataframe().columns]
    raw_df = pd.read_csv(RAW_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    _validate_frame("raw data", raw_df, expected_columns)
    _validate_frame("training data", train_df, expected_columns)
    _validate_frame("test data", test_df, expected_columns)

    if len(train_df) <= len(test_df):
        raise ValueError("training split must be larger than test split")

    feature_count = len(expected_columns) - 2
    report = {
        "valid": 1,
        "raw_rows": int(len(raw_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(feature_count),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_VALIDATION_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return ValidationResult(
        report_path=str(DATA_VALIDATION_PATH),
        raw_rows=report["raw_rows"],
        train_rows=report["train_rows"],
        test_rows=report["test_rows"],
        feature_count=report["feature_count"],
        valid=bool(report["valid"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared tabular datasets.")
    parser.parse_args()
    result = validate_data()
    print(f"Data validation report saved to {result.report_path}")
    print(
        json.dumps(
            {
                "valid": int(result.valid),
                "raw_rows": result.raw_rows,
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
                "feature_count": result.feature_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
