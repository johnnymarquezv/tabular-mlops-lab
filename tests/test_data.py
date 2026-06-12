from __future__ import annotations

from mlops_tabular.data import (
    TARGET_COLUMN,
    TARGET_NAME_COLUMN,
    load_demo_dataframe,
    load_processed_data,
    prepare_data,
)


def test_load_demo_dataframe_has_expected_schema() -> None:
    dataframe = load_demo_dataframe()

    assert not dataframe.empty
    assert TARGET_COLUMN in dataframe.columns
    assert TARGET_NAME_COLUMN in dataframe.columns
    assert dataframe[TARGET_COLUMN].nunique() == 2
    assert "mean_radius" in dataframe.columns


def test_prepare_data_writes_train_and_test_splits() -> None:
    paths = prepare_data()
    train_df, test_df = load_processed_data()

    assert paths.raw.exists()
    assert paths.train.exists()
    assert paths.test.exists()
    assert len(train_df) > len(test_df)
    assert set(train_df[TARGET_COLUMN].unique()) == set(test_df[TARGET_COLUMN].unique())
