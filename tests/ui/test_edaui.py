import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 100],
        "salary": [50000, 60000, 70000, 80000, 90000, 120000],
        "category": ["A", "B", "A", "B", "C", "A"],
        "date_col": pd.date_range("2026-01-01", periods=6, freq="D").strftime("%Y-%m-%d")
    })

def run_app_with_df(df=None, **kwargs):
    def script_entrypoint():
        from smart_analytics.ui.eda_ui import smartedaui
        # Instantiate and execute call to ensure _sidebar_ux is run
        app = smartedaui(df=df, **kwargs)
        if df is not None:
            app()

    at = AppTest.from_function(script_entrypoint)
    at.run()
    return at


# --- TEST CASES ---

def test_initial_state_without_data():
    """Verify page behavior when no dataframe is uploaded or passed."""
    def entrypoint():
        from smart_analytics.ui.eda_ui import smartedaui
        smartedaui()()

    at = AppTest.from_function(entrypoint).run()
    
    assert len(at.title) > 0
    assert "Provide a dataframe" in at.title[0].value


def test_rendering_with_data(sample_df):
    """Verify that title and primary sidebar controls render properly when data is passed."""
    at = run_app_with_df(sample_df)
    
    # Verify Main Title
    assert at.title[0].value == "SmartEda report"
    
    # Check key-indexed sidebar widgets
    assert at.checkbox(key="Visualize numerical").value is False
    assert at.checkbox(key="Visualize categorical").value is False
    assert at.checkbox(key="Visualize heatmap").value is False
    assert at.selectbox(key="Index column").value is None
    assert at.selectbox(key="Date column").value is None


def test_toggle_numerical_visualizations(sample_df):
    """Verify that checking 'Visualize numerical' renders Plotly histograms."""
    at = run_app_with_df(sample_df)
    
    # Initial state: no plotly charts
    assert len(at.plotly_chart) == 0 if hasattr(at, 'plotly_chart') else len(at.get("plotly_chart")) == 0
    
    # Toggle 'Visualize numerical' using explicit key
    at.checkbox(key="Visualize numerical").check().run()
    
    # Expect 2 histograms rendered (for 'age' and 'salary')
    assert len(at.get("plotly_chart")) == 2


def test_box_plot_toggle(sample_df):
    """Verify that enabling box plots doubles the rendered Plotly charts for numerical features."""
    at = run_app_with_df(sample_df)
    
    # Enable both numerical vis and box plots
    at.checkbox(key="Visualize numerical").check().run()
    at.checkbox(key="Visualize box").check().run()
    
    # 2 histograms + 2 box plots = 4 charts total
    assert len(at.get("plotly_chart")) == 4


def test_toggle_categorical_visualizations(sample_df):
    """Verify that checking 'Visualize categorical' renders bar charts."""
    at = run_app_with_df(sample_df)
    
    at.checkbox(key="Visualize categorical").check().run()
    
    # Expect 1 bar chart for the 'category' column
    assert len(at.get("plotly_chart")) >= 1


def test_toggle_heatmap_visualization(sample_df):
    """Verify that checking 'Visualize heatmap' renders the correlation matrix."""
    at = run_app_with_df(sample_df)
    
    at.checkbox(key="Visualize heatmap").check().run()
    
    # Expect 1 correlation heatmap
    assert len(at.get("plotly_chart")) == 1


def test_advanced_options_sub_widgets(sample_df):
    """Verify that enabling 'Advanced Options' displays dependent dropdowns."""
    at = run_app_with_df(sample_df)
    
    # Enable Advanced Options toggle (matches exact label/key structure)
    at.checkbox[1].check().run()  # Selects 'Advanced Options' checkbox inside configurations
    
    # Check that advanced histogram selectboxes appear
    assert at.selectbox(key="Histogram color method").value is None
    assert at.selectbox(key="Histogram bar method").value == "relative"
    assert at.selectbox(key="Histogram marginal method").value is None


def test_custom_date_format_input(sample_df):
    """Verify that selecting 'Custom format' displays the text input widget."""
    at = run_app_with_df(sample_df)
    
    # Select Custom Format in the dropdown
    at.selectbox(key="Date format").select("Custom format (Enter below)").run()
    
    # Verify custom date format text input appears with default value
    assert at.text_input(key="Custom date format").value == "%Y-%m-%d"
    
    # Change format text value
    at.text_input(key="Custom date format").input("%d/%m/%Y").run()
    assert at.text_input(key="Custom date format").value == "%d/%m/%Y"


def test_html_download_button_presence(sample_df):
    """Verify that the HTML download button exists and is configured correctly."""
    at = run_app_with_df(sample_df)
    
    # Verify key-indexed download button
    download_btn = at.down(key="html download button")
    assert download_btn.label == "Download EDA HTML Report"