from smart_analytics.core.smarteda import SmartEDA
import pytest
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
import json

# ==========================================
# Tests for Initialization & Core Data Types
# ==========================================

def test_smarteda_init_with_dataframe():
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": ["something", "something"]})
    eda = SmartEDA(df=df)

    assert isinstance(eda.df, pd.DataFrame)
    assert isinstance(eda.all_cols, list)
    assert "col1" in eda.numeric_cols
    assert "col2" in eda.numeric_cols
    assert "col3" in eda.categorical_cols

def test_smarteda_init_json(tmp_path):
    json_file = tmp_path / "test_data.json"
    json_file.write_text(json.dumps({"id": [101, 202], "status": ["ok", "fail"], "date": ["2024/01/01", "2024/01/02"]}))
    eda = SmartEDA(df=json_file, date_column="date", date_format="%Y/%m/%d")

    assert "id" in eda.numeric_cols
    assert "date" in eda.all_cols
    assert "status" in eda.categorical_cols
    assert "date" not in eda.numeric_cols
    assert "date" not in eda.categorical_cols

def test_smarteda_init_with_io():
    csv_data = b"col1,col2,date\n1,something,01/01/2024\n3,something,02/01/2024"
    bytes_io = io.BytesIO(csv_data)
    eda = SmartEDA(df=bytes_io, date_column="date", date_format="%d/%m/%Y")

    assert isinstance(eda.df, pd.DataFrame)
    assert isinstance(eda.all_cols, list)
    assert "col1" in eda.numeric_cols
    assert "col2" in eda.categorical_cols
    assert "date" in eda.all_cols
    assert "date" not in eda.numeric_cols
    assert "date" not in eda.categorical_cols
    assert pd.api.types.is_datetime64_any_dtype(eda.df["date"])


def test_eda_initialization_plots():
    df = pd.DataFrame({"col1": [1, 2], "date": ["01/01/2024", "02/01/2024"], "col3": ["something", "something"]})
    eda = SmartEDA(df=df, save_numerical_figures=True, date_column="date", date_format="%d/%m/%Y")

    assert len(eda.numerical_hist_list) == 1
    assert isinstance(eda.numerical_hist_dict["col1"], go.Figure)
    assert len(eda.categorical_bar_list) == 0
    assert pd.api.types.is_datetime64_any_dtype(eda.df["date"])


# ==========================================
# New Comprehensive Tests
# ==========================================

def test_invalid_iqr_method_warning():
    df = pd.DataFrame({"col1": [1, 2, 3, 100]})
    with pytest.warns(UserWarning, match="Invalid method"):
        eda = SmartEDA(df=df, handle_iqr="invalid_method")
    assert eda.handle_iqr == "ignore"


def test_iqr_nan_transformation():
    # Outliers should be converted to NaN when handle_iqr="nan"
    df = pd.DataFrame({"col1": [10, 12, 11, 13, 1000]})
    eda = SmartEDA(df=df, handle_iqr="nan")
    
    assert pd.isna(eda.df["col1"].iloc[-1])
    assert not pd.isna(eda.df["col1"].iloc[0])


def test_index_column_assignment():
    df = pd.DataFrame({"id_col": ["a", "b"], "col1": [1, 2], "col2": [3, 4]})
    eda = SmartEDA(df=df, index_column="id_col")

    assert eda.df.index.name == "id_col"
    assert "id_col" not in eda.numeric_cols
    assert "id_col" not in eda.categorical_cols


def test_save_px_html_generation():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    eda = SmartEDA(
        df=df, 
        save_numerical_figures=True, 
        save_categorical_figures=True, 
        save_heatmap_figure=True,
        dataset_name="test_dataset"
    )
    
    html_output = eda._save_px_html()
    assert isinstance(html_output, str)
    assert "<title>EDA Results</title>" in html_output
    assert "test_dataset" in html_output
    assert "plotly" in html_output


def test__str__method_representation():
    df = pd.DataFrame({"col1": [1, 2], "col2": ["x", "y"]})
    eda = SmartEDA(df=df, dataset_name="string_test")
    
    str_repr = str(eda)
    assert "string_test" in str_repr
    assert "Dataset details:" in str_repr


def test_callable_instance():
    df1 = pd.DataFrame({"col1": [1, 2]})
    df2 = pd.DataFrame({"col2": [10, 20]})
    
    eda = SmartEDA(df=df1)
    assert "col1" in eda.numeric_cols
    
    eda(df=df2)
    assert "col2" in eda.numeric_cols
    assert "col1" not in eda.numeric_cols

def test_getitem_method():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    eda = SmartEDA(df=df)

    assert eda[0] == "col1"
    assert eda[1] == "col2"

def test_len_method():
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    eda = SmartEDA(df=df)

    assert len(eda) == 2