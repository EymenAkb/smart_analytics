# smart_analytics

This repository was developed by **EymenAkb** as a multi-dashboard analytics tool utilizing **Plotly** and **Streamlit**.

An Object-Oriented Programming (OOP) Python library designed to streamline Exploratory Data Analysis (EDA) and Time Series analytics. It automates data handling, summary generation, and interactive visualization using **Plotly**, complete with a built-in **Streamlit** dashboard application for rapid web-based exploration and HTML report extraction.

---

## Key Components

The package is structured into robust core analytics pipelines and modular user interfaces:

* **`Data`**: Automated data handler class managing data loading, missing values, custom date parsing, indexing, and IQR outlier handling.
* **`SmartEDA`**: Automated exploratory data analysis engine for numerical, categorical, and correlation visual summaries.
* **`smartedaui` / UI Modules**: Streamlit-powered dashboard creator for multi-page real-time exploratory analytics and interactive configuration.
* **`SmartTimeSeries`**: Automated time series analysis for numerical visualizations *(currently in prototype phase; categorical support under development)*.
* **`tsui`**: Streamlit-based dashboard version of the time series engine.

---

## Project Structure

```text
smart_analytics/
│
├── .gitattributes
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── src/
    └── smart_analytics/
        ├── __init__.py
        ├── core/
        │   ├── __init__.py
        │   ├── data.py
        │   ├── smarteda.py
        │   └── timeseries.py
        └── ui/
            ├── __init__.py
            ├── app.py
            ├── eda_ui.py
            ├── mainpage.py
            └── time_ui.py

```

---

## Tech Stack & Dependencies

* **Pandas**: Data ingestion, cleaning, and tabular manipulation.
* **NumPy**: Numerical operations and mathematical calculations.
* **Plotly**: Interactive, publication-ready graphical visualizations.
* **Streamlit**: Dynamic, multi-page dashboard interface rendering.

---

## Installation & Setup

Install the package locally in your environment using pip:

```bash
pip install .

```

---

## Usage Guide

### 1. Using the Automated Data Handler and EDA Core

```python
import pandas as pd
from smart_analytics.core.data import Data
from smart_analytics.core.smarteda import SmartEDA

# Load data automatically using the built-in handler
df = Data.load_data("your_dataset.csv", date_col="date_column")

# Run automated EDA with visualization parameters
eda_report = SmartEDA(
    df=df,
    visualize_numerical=True,
    visualize_categorical=True,
    visualize_heatmap=True,
    dataset_name="sales_data",
    handle_iqr="nan"
)

# Print summary information
print(eda_report)

```

### 2. Launching the Streamlit Dashboard UI

Launch the interactive Streamlit dashboard interface directly from your command line using the registered console script:

```bash
launch-analytics

```

Alternatively, run the app directly via Streamlit:

```bash
streamlit run src/smart_analytics/ui/app.py

```

---

## Features & Methods Reference

### `Data` Class

* **`load_data`**: Dynamically accepts file paths (`str`), `pd.DataFrame`, or binary streams (`io.BytesIO`), safely applying date and index configurations.
* **`column_assigner`**: Automatically separates and categorizes columns into numerical, categorical, and date variables.
* **`return_iqr`**: Filters out outlier values using Interquartile Range (IQR) calculations.

### `SmartEDA` Class

* Generates interactive Plotly histograms, box plots, bar charts, and correlation heatmaps.
* Supports raw HTML report generation (`_save_px_html`) with embedded Plotly scripts for offline sharing.

### UI Modules (`ui/`)

* Integrates Streamlit components (`app.py`, `eda_ui.py`, `mainpage.py`, `time_ui.py`) to deliver a modular, multi-page user interface equipped with file uploaders, mapping choices, custom date format parsers, and report downloading options.

---

> [!NOTE]
> Ensure your date columns adhere to standard formats or use the custom format builder inside the Streamlit UI for seamless parsing.
> The program is currently on the prototype phase, results might include some inaccuracies.