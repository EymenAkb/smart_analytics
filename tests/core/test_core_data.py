from __future__ import annotations

import io
import json
import os
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import pytest

from smart_analytics.core.data import Data

# ==========================================
# TESTS FOR load_data
# ==========================================

def test_load_data_with_dataframe():
    """Passing pd.DataFrame into Data class returns a copy."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    result_df = Data.load_data(df=df)

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
    assert result_df is not df


def test_load_data_with_valid_input_when_can_return_none_is_true():
    """Valid DataFrame with can_return_none=True returns a valid dataframe."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    result_df = Data.load_data(df=df, can_return_none=True)

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2
    assert result_df is not df
    assert not result_df.empty


def test_load_data_with_bytes_io():
    """Passing BytesIO object loads correctly."""
    csv_data = b"col1,col2\n1,3\n2,4"
    bytes_io = io.BytesIO(csv_data)
    result_df = Data.load_data(df=bytes_io)

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["col1", "col2"]
    assert len(result_df) == 2


def test_load_data_invalid_input_raises_error():
    """Invalid input type without can_return_none raises ValueError."""
    with pytest.raises(ValueError, match="Invalid or missing dataframe input."):
        Data.load_data(df=12345, can_return_none=False)


def test_load_data_invalid_input_returns_none():
    """Invalid input with can_return_none=True warns and returns None."""
    with pytest.warns(UserWarning):
        result = Data.load_data(df=12345, can_return_none=True)
        assert result is None


def test_load_data_json(tmp_path):
    json_file = tmp_path / "test_data.json"
    json_file.write_text(
        json.dumps({"id": [101, 202], "status": ["ok", "fail"]})
    )
    df, numerical_columns, categorical_columns, all_cols = Data.load_data(
        json_file, return_columns=True
    )

    assert isinstance(df, pd.DataFrame)
    assert isinstance(numerical_columns, list)
    assert "id" in numerical_columns
    assert isinstance(categorical_columns, list)
    assert "status" in categorical_columns
    assert isinstance(all_cols, list)
    assert "id" in all_cols
    assert "status" in all_cols


# ==========================================
# TESTS FOR column_assigner & date_check
# ==========================================


def test_column_assigner():
    """Correctly categorizes columns and excludes date columns from regular lists."""
    df = pd.DataFrame(
        {
            "num_col": [1, 2, 3],
            "cat_col": ["a", "b", "c"],
            "date_col": ["2026-01-01", "2026-01-02", None],
        }
    )

    df["date_col"] = pd.to_datetime(df["date_col"], format="%Y-%m-%d")

    num, cat, all_cols = Data.column_assigner(data=df, date_column="date_col")

    assert "num_col" in num
    assert "cat_col" in cat
    assert "date_col" in all_cols
    assert "date_col" not in num
    assert "date_col" not in cat
    assert "num_col" in all_cols
    assert "cat_col" in all_cols


def test_column_assigner_invalid_data():
    """Passing non-dataframe to column_assigner raises ValueError."""
    with pytest.raises(
        ValueError, match="Provided dataframe is not a pd.DataFrame object."
    ):
        Data.column_assigner(data="not_a_dataframe")


# ==========================================
# TESTS FOR assign_index
# ==========================================


def test_assign_index_by_string():
    """Successfully sets a dataframe index using column name string."""
    df = pd.DataFrame({"id": ["a", "b"], "val": [1, 2]})
    result_df = Data.assign_index(df, index_column="id")

    assert result_df.index.name == "id"


def test_assign_index_by_int():
    """Successfully sets dataframe index using column position integer."""
    df = pd.DataFrame({"id": ["a", "b"], "val": [1, 2]})
    result_df = Data.assign_index(df, index_column=0)

    assert result_df.index.name == "id"


def test_assign_index_by_list():
    df = pd.DataFrame({"id": [1, 2], "id2": [3, 4]})
    result_df = Data.assign_index(data=df, index_column=["id", "id2"])

    assert "id" in result_df.index.names
    assert "id2" in result_df.index.names


def test_assign_index_single_element_list():
    """Catch the unboxing bug for single-item lists."""
    df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    res = Data.assign_index(df, index_column=["id"])
    assert res.index.name == "id"


def test_assign_index_by_tuple_and_set():
    """Verify tuples and sets set multi-level index properly."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "val": [10, 20]})

    res_tuple = Data.assign_index(df.copy(), index_column=("a", "b"))
    assert res_tuple.index.names == ["a", "b"]

    res_set = Data.assign_index(df.copy(), index_column={"b", "a"})
    assert res_set.index.names == ["a", "b"]


# ==========================================
# TESTS FOR calculate_iqr & return_iqr
# ==========================================


def test_calculate_iqr():
    """Calculates low and high IQR thresholds correctly."""
    series = pd.Series([1, 2, 3, 4, 5, 100])
    low, high = Data.calculate_iqr(series)

    assert isinstance(low, (int, float, np.number))
    assert isinstance(high, (int, float, np.number))
    assert low < high


def test_return_iqr_masks_outliers():
    """Masks outliers beyond IQR boundaries to NaN."""
    df = pd.DataFrame({"val": [10, 11, 12, 13, 14, 1000]})
    result_df = Data.return_iqr(df, columns="val")

    assert pd.isna(result_df["val"].iloc[-1])
    assert result_df["val"].iloc[0] == 10


def test_return_iqr_invalid_dataframe():
    """Passing non-dataframe to return_iqr raises ValueError."""
    with pytest.raises(ValueError, match="DataFrame isn't a pandas DataFrame."):
        Data.return_iqr("not_a_df", columns=["val"])


def test_return_iqr_missing_column_warning():
    """Passing non-existent column triggers a warning."""
    df = pd.DataFrame({"val": [1, 2, 3]})
    with pytest.raises(ValueError):
        Data.return_iqr(df, columns=["missing_col"])