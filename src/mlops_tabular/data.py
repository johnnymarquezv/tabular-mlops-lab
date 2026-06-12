"""Dataset creation and loading helpers."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mlops_tabular.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, Settings, settings

RAW_DATA_PATH = RAW_DATA_DIR / "breast_cancer.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
TARGET_COLUMN = "target"
TARGET_NAME_COLUMN = "target_name"


@dataclass(frozen=True)
class DataPaths:
    """Filesystem locations produced by the data preparation step."""

    raw: Path
    train: Path
    test: Path


def sanitize_feature_name(name: str) -> str:
    """Convert sklearn feature labels into API-friendly snake_case names."""

    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def load_demo_dataframe() -> pd.DataFrame:
    """Load the built-in breast cancer dataset as a clean dataframe."""

    dataset: Any = load_breast_cancer(as_frame=True)
    features = cast(pd.DataFrame, dataset.data).copy()
    target = cast(pd.Series, dataset.target).astype(int)
    target_names = [str(value) for value in dataset.target_names]
    feature_names = [sanitize_feature_name(str(column)) for column in features.columns]

    features.columns = feature_names
    features[TARGET_COLUMN] = target
    features[TARGET_NAME_COLUMN] = features[TARGET_COLUMN].map(
        lambda label: target_names[int(label)]
    )
    return features


def prepare_data(active_settings: Settings = settings) -> DataPaths:
    """Write raw and train/test CSV files for reproducible local workflows."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_demo_dataframe()
    dataframe.to_csv(RAW_DATA_PATH, index=False)

    train_df, test_df = train_test_split(
        dataframe,
        test_size=active_settings.test_size,
        random_state=active_settings.random_state,
        stratify=dataframe[TARGET_COLUMN],
    )

    cast(pd.DataFrame, train_df).to_csv(TRAIN_DATA_PATH, index=False)
    cast(pd.DataFrame, test_df).to_csv(TEST_DATA_PATH, index=False)

    return DataPaths(raw=RAW_DATA_PATH, train=TRAIN_DATA_PATH, test=TEST_DATA_PATH)


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prepared train and test splits, creating them if needed."""

    if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
        prepare_data()

    return pd.read_csv(TRAIN_DATA_PATH), pd.read_csv(TEST_DATA_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the demo tabular dataset.")
    parser.parse_args()
    paths = prepare_data()
    print(f"Wrote raw data to {paths.raw}")
    print(f"Wrote training data to {paths.train}")
    print(f"Wrote test data to {paths.test}")


if __name__ == "__main__":
    main()
