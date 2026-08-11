import io
import warnings
import numpy as np
import pandas as pd
from smart_analytics.core.data import Data
import pytest

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

# ==========================================
# TESTS FOR column_assigner & date_check
# ==========================================

def test_column_assigner():
    """Correctly categorizes columns and excludes date columns from regular lists."""
    df = pd.DataFrame({
        "num_col": [1, 2, 3],
        "cat_col": ["a", "b", "c"],
        "date_col": ["2026-01-01", "2026-01-02", None]
    })
    
    df["date_col"] = pd.to_datetime(df["date_col"], format="%Y-%m-%d")

    num, cat, all_cols = Data.column_assigner(data=df, date_column="date_col")

    assert "num_col" in num
    assert "cat_col" in cat
    assert "date_col" not in all_cols
    assert "num_col" in all_cols
    assert "cat_col" in all_cols

def test_column_assigner_invalid_data():
    """Passing non-dataframe to column_assigner raises ValueError."""
    with pytest.raises(ValueError, match="Provided dataframe is not a pd.DataFrame object."):
        Data.column_assigner(data="not_a_dataframe")

def test_date_check():
    """Correctly identifies datetime columns."""
    df = pd.DataFrame({
        "dt": pd.to_datetime(["2026-01-01"]),
        "not_dt": [123]
    })
    assert Data.date_check(df, "dt") is True
    assert Data.date_check(df, "not_dt") is False
    assert Data.date_check(df, "missing") is False

# ==========================================
# TESTS FOR assign_date & handle_date_assignment
# ==========================================

def test_assign_date_success():
    """Successfully converts string column to datetime."""
    df = pd.DataFrame({"date_col": ["01/01/2026", "02/01/2026"]})
    result_df = Data.assign_date(df, "date_col", date_format="%d/%m/%Y")
    
    assert pd.api.types.is_datetime64_any_dtype(result_df["date_col"])

def test_assign_date_invalid_column_warning():
    """Invalid date format or column triggers warning and returns original dataframe."""
    df = pd.DataFrame({"num_col": [1, 2]})
    with pytest.warns(UserWarning, match="Could not assign the column"):
        result_df = Data.assign_date(df, "non_existent_col")
        assert result_df is not None

def test_handle_date_assignment():
    """Handles successful date setup and returns columns when requested."""
    df = pd.DataFrame({
        "val": [10, 20],
        "date": ["2026-01-01", "2026-01-02"]
    })
    res_df, num, cat, all_cols = Data.handle_date_assignment(
        data=df, date_column="date", date_format="%Y-%m-%d", return_columns=True
    )
    
    assert pd.api.types.is_datetime64_any_dtype(res_df["date"])
    assert "val" in num
    assert "date" not in all_cols

# ==========================================
# TESTS FOR assign_index & handle_index_assignment
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

def test_handle_index_assignment_with_return_columns():
    """Handles index assignment and accurately returns updated column categories."""
    df = pd.DataFrame({"id": ["a", "b"], "num": [10, 20], "cat": ["x", "y"]})
    res_df, num, cat, all_cols = Data.handle_index_assignment(
        data=df, index_column="id", return_columns=True
    )
    
    assert res_df.index.name == "id"
    assert "num" in num
    assert "cat" in cat
    assert "id" not in all_cols

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
    with pytest.warns(UserWarning, match="Column 'missing_col' not found"):
        Data.return_iqr(df, columns=["missing_col"])