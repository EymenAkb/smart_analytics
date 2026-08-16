from smart_analytics.core.timeseries import SmartTimeSeries
import pytest
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go
import json

# ==========================================
# Tests for Initialization & Core Data Types
# ==========================================

def test_time_series_init_dataframe():
    df = pd.DataFrame({"date": ["2024/01/01", "2024/01/02"], "num": [1, 2], "cat": ["object", "something"]})
    timeseries = SmartTimeSeries(df=df, date_column="date", date_format="%Y/%m/%d")

    assert "date" in timeseries.all_columns
    assert "num" in timeseries.numerical_columns
    assert "cat" in timeseries.categorical_columns
    assert "date" not in timeseries.categorical_columns
    assert "date" not in timeseries.numerical_columns

def test_timeseries_init_json(tmp_path):
    json_file = tmp_path / "test_data.json"
    json_file.write_text(json.dumps({"id": [101, 202], "status": ["ok", "fail"], "date": ["2024/01/01", "2024/01/02"]}))
    timeseries = SmartTimeSeries(df=json_file, date_column="date", date_format="%Y/%m/%d")

    assert "date" in timeseries.all_columns
    assert "id" in timeseries.numerical_columns
    assert "status" in timeseries.categorical_columns
    assert "date" not in timeseries.categorical_columns
    assert "date" not in timeseries.numerical_columns

def test_timeseries_init_io():
    csv_data = b"col1,col2,date\n1,something,01/01/2024\n3,something,02/01/2024"
    obj = io.BytesIO(csv_data)
    timeseries = SmartTimeSeries(df=obj, date_column="date", date_format="%d/%m/%Y")

    assert "date" in timeseries.all_columns
    assert "col1" in timeseries.numerical_columns
    assert "col2" in timeseries.categorical_columns
    assert "date" not in timeseries.categorical_columns
    assert "date" not in timeseries.numerical_columns

def test_timeseries_init_plots():
    df = pd.DataFrame({"date": ["2024/01/01", "2024/01/02"], "num": [1, 2], "cat": ["object", "something"]})
    timeseries = SmartTimeSeries(df=df, date_column="date", date_format="%Y/%m/%d", visualize_area_graph=True, save_area_graphs=True,
                                 visualize_line_categorical=True, visualize_line_graph_numerical=True, visualize_scatter=True, visualize_treemap=True,
                                 save_line_categorical=True, save_line_graphs_numerical=True, save_scatter_graphs=True, save_treemap=True)

    assert isinstance(timeseries.area_figures[0], go.Figure)
    assert isinstance(timeseries.treemap_figures[0], go.Figure)
    assert isinstance(timeseries.numerical_line_figures[0], go.Figure)
    assert isinstance(timeseries.categorical_line_figures[0], go.Figure)
    assert isinstance(timeseries.scatter_figures[0], go.Figure)

# ==========================================
# Tests for magic (dunder) methods
# ==========================================

def test_str_method():
    df = pd.DataFrame({"date": ["2024/01/01", "2024/01/02"], "num": [1, 2], "cat": ["object", "something"]})
    timeseries = SmartTimeSeries(df=df, date_column="date", date_format="%Y/%m/%d", visualize_area_graph=True, save_area_graphs=True,
    visualize_line_categorical=True, visualize_line_graph_numerical=True, visualize_scatter=True, visualize_treemap=True,
    save_line_categorical=True, save_line_graphs_numerical=True, save_scatter_graphs=True, save_treemap=True)

    result = str(timeseries)

    assert "num" in result
    assert "cat" in result
    assert "date" in result
    assert "Dataset details:" in result

def test_get_item_method():
    df = pd.DataFrame({"date": ["2024/01/01", "2024/01/02"], "num": [1, 2], "cat": ["object", "something"]})
    timeseries = SmartTimeSeries(df=df)

    assert timeseries[0] == "date"
    assert timeseries[1] == "num"
    assert timeseries[2] == "cat"

def test_len_method():
    df = pd.DataFrame({"date": ["2024/01/01", "2024/01/02"], "num": [1, 2], "cat": ["object", "something"]})
    timeseries = SmartTimeSeries(df=df)

    assert len(timeseries) == 3