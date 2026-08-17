from smart_analytics.core.smarteda import SmartEDA
from smart_analytics.core.data import Data
import streamlit as st
from typing import Literal, get_args
import pandas as pd
import io
from pathlib import Path
import warnings

IQRHandleMethod = Literal["ignore", "nan"]
HistogramMarginalValues = Literal[None, "box", "rug", "violin"]
HistogramBarModes = Literal["relative", "overlay", "group", "stack"]

@st.cache_data
def load_data(df: str | pd.DataFrame | io.BytesIO | Path | None = None, index_col: str | int | list | set | tuple | None = None, 
                  date_col: str | int | None = None, date_format="mixed", can_return_none: bool = False, 
                  return_columns: bool = False) -> None | pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
    return Data.load_data(df=df, index_col=index_col, date_col=date_col, date_format=date_format, can_return_none=can_return_none, return_columns=return_columns)

class smartedaui(SmartEDA):
    DATE_FORMATS = {
        "YYYY-MM-DD (e.g., 2026-06-06)": "%Y-%m-%d",
        "YYYY-MM-DD HH:MM:SS (e.g., 2026-06-06 14:30:00)": "%Y-%m-%d %H:%M:%S",
        "MM/DD/YYYY (US, e.g., 06/06/2026)": "%m/%d/%Y",
        "DD/MM/YYYY (UK/EU, e.g., 06/06/2026)": "%d/%m/%Y",
        "DD/MM/YY (Short UK, e.g., 06/06/26)": "%d/%m/%y",
        "DD.MM.YYYY (e.g., 06.06.2026)": "%d.%m.%Y",
        "DD.MM.YY (e.g., 06.06.26)": "%d.%m.%y",
        "Month DD, YYYY (e.g., June 06, 2026)": "%B %d, %Y",
        "DD Month YYYY (e.g., 06 June 2026)": "%d %B %Y",
        "Mon DD, YYYY (e.g., Jun 06, 2026)": "%b %d, %Y",
        "DD-Mon-YYYY (e.g., 06-Jun-2026)": "%d-%b-%Y",
        "MM/DD/YYYY HH:MM AM/PM (e.g., 06/06/2026 02:30 PM)": "%m/%d/%Y %I:%M %p",
        "Custom format (Enter below)": "custom"
    }
    def __init__(self,
        df: pd.DataFrame | str | io.BytesIO = None,
        visualize_numerical: bool = False,
        visualize_categorical: bool = False,
        visualize_heatmap: bool = False,
        save_numerical_figures: bool = False,
        histogram_marginal: HistogramMarginalValues = None,
        save_categorical_figures: bool = False,
        save_heatmap_figure: bool = False,
        show_info: bool = False,
        dataset_name:str = "dataset",
        handle_iqr: IQRHandleMethod = "ignore",
        histogram_color_method: bool = False,
        histogram_bar_mode: HistogramBarModes = "relative",
        show_iqr_box: bool = False,
        date_column: str | int | None = None,
        date_format: str = None,
        index_column: str | int | None = None):
        self.iqr_methods = get_args(IQRHandleMethod)
        self.hist_marginal_methods = get_args(HistogramMarginalValues)
        self.hist_bar_modes = get_args(HistogramBarModes)
        handle_iqr = handle_iqr.lower().strip()

        if "advanced_options" not in st.session_state:
            st.session_state["advanced_options"] = False

        if not handle_iqr in self.iqr_methods:
            warnings.warn(f"Invalid method '{handle_iqr}'. Expected one of {self.iqr_methods}, falling back to 'ignore'.", category=UserWarning)
            handle_iqr = "ignore"

        if not histogram_marginal in self.hist_marginal_methods:
            warnings.warn(f"Invalid marginal value: {histogram_marginal}. Expected one of: {self.hist_marginal_methods}, falling back to None", category=UserWarning)
            histogram_marginal = None

        if not histogram_bar_mode in self.hist_bar_modes:
            warnings.warn(f"Invalid bar mode value: {histogram_marginal}. Expected one of: {self.hist_bar_modes}, falling back to overlay", category=UserWarning)
            histogram_marginal = "relative"

        self.histogram_color_method = histogram_color_method
        self.histogram_bar_mode = histogram_bar_mode
        self.histogram_marginal = histogram_marginal
        self.handle_iqr = handle_iqr
        self.visnum = visualize_numerical
        self.viscat = visualize_categorical
        self.vishm = visualize_heatmap
        self.show_info = show_info
        self.save_numerical = save_numerical_figures
        self.save_categorical = save_categorical_figures
        self.save_heatmap = save_heatmap_figure
        self.dataset = dataset_name
        self.show_iqr_box = show_iqr_box
        self.date_column = date_column
        self.date_format = date_format
        self.index_column = index_column

        self.df = load_data(df=df, index_col=index_column, date_col=date_column, date_format=date_format, can_return_none=True, return_columns=False)

        if isinstance(self.df, pd.DataFrame):
            self._run()

    def _run(self):
        if isinstance(self.df, pd.DataFrame):
            self._create_saving_lists()
            self.numeric_cols, self.categorical_cols, self.all_cols = Data.column_assigner(data=self.df, date_column=self.date_column)
            st.title("SmartEda report")
            st.sidebar.divider()
            self._create_assignments_ux()
            self._apply_transformations()
            self.numeric_cols, self.categorical_cols, self.all_cols = Data.column_assigner(data=self.df, date_column=self.date_column)
            self._create_render_plots()
            self._html_extract_ux()
            self._html_download_ux()
        else:
            st.title(f"Provide a dataframe in the sidebar to continue.")


    def _create_render_plots(self):
        if self.show_info:
            self.info_show()

        if (self.visnum or self.save_numerical) and isinstance(self.numeric_cols, list):
            self._create_numerical_visualization()

        if (self.viscat or self.save_categorical) and isinstance(self.categorical_cols, list):
            self._create_categorical_vis()

        if (self.vishm or self.save_heatmap) and isinstance(self.numeric_cols, list):
            self._create_heatmap_figure()

        if (self.save_categorical or self.save_heatmap or self.save_numerical) and isinstance(self.numeric_cols, list) and isinstance(self.categorical_cols, list):
            self._save_px_html()

    def _create_numerical_visualization(self):
        if not self.numeric_cols:
            return
        st.header("Numerical visualizations")
        c1, c2 = st.columns([2, 1] if self.show_iqr_box else [1, 1])

        for i, column in enumerate(self.numeric_cols):
            if self.show_iqr_box:
                with c1:
                    st.plotly_chart(self._create_histogram_visualization(column=column, marginal=self.histogram_marginal, 
                                                                         barmode=self.histogram_bar_mode, color=self.histogram_color_method), width="stretch")
                with c2:
                    st.plotly_chart(self._create_box_plot(column=column), width="stretch")
            else:
                target = c1 if i % 2 == 0 else c2
                with target:
                    st.plotly_chart(self._create_histogram_visualization(column=column, marginal=self.histogram_marginal, 
                                                                         barmode=self.histogram_bar_mode, color=self.histogram_color_method), width="stretch")

    def _create_categorical_vis(self):
        if not self.categorical_cols:
            return
        
        st.header("Categorical visualizations")
        c1, c2 = st.columns([1,1])

        for i, column in enumerate(self.categorical_cols):
            target = c1 if not i % 2 else c2
            with target:
                st.plotly_chart(self._create_bar_figure(column=column), width="stretch")

    def _create_heatmap_figure(self):
        if not self.numeric_cols:
            return
        
        st.header("Correlation heatmap")
        st.plotly_chart(self._create_heatmap_vis(), width="stretch")

    def _df_initilazier(self):
        st.sidebar.header("Data Downloader")
        df = st.sidebar.file_uploader("Upload csv file", type=["csv"])
        if df:
            self.df = load_data(df)

    def _create_numerical_ux(self):
        st.sidebar.header("Numerical Options")
        self.visnum = st.sidebar.checkbox("Visualize numerical", value=self.visnum)
        if st.session_state["advanced_options"]:
            st.sidebar.text("Advanced Histogram Options")
            self.histogram_color_method = st.sidebar.selectbox("Histogram color method", options=[None, "Column"])
            self.histogram_bar_mode = st.sidebar.selectbox("Histogram bar mode", options=self.hist_bar_modes)
            self.histogram_marginal = st.sidebar.selectbox("Histogram marginal", options=self.hist_marginal_methods)
        self.show_iqr_box = st.sidebar.checkbox("Show box graph", value=self.show_iqr_box,
            help="If handle iqr is set to 'nan' the box plots might show wrong details!")
        self.vishm = st.sidebar.checkbox("Visualize heatmap", value=self.vishm)
        self.handle_iqr = st.sidebar.selectbox("Handle iqr method", get_args(IQRHandleMethod),
            help="If handle iqr is set to 'nan' the box plots might show wrong details!")
        
    def _create_categorical_ux(self):
        st.sidebar.header("Categorical Options")
        self.viscat = st.sidebar.checkbox("Visualize categorical", value=self.viscat)

    def _create_assignments_ux(self):
        st.sidebar.header("Index and date assignment")
        self.index_column = st.sidebar.selectbox("Index column", options= [None] + self.all_cols)
        self.date_column = st.sidebar.selectbox("Date column", options= [None] + self.all_cols)

        selected_label = st.sidebar.selectbox("Select Date Format", options=list(self.DATE_FORMATS.keys()))

        if selected_label == "Custom format (Enter below)":
            self.chosen_format = st.sidebar.text_input(
            "Enter Custom Format String",
            value="%Y-%m-%d",
            help="Specify format directives starting with %. Example: %Y-%m-%d for 2026-06-06. Characters between them act as separators."
            )
            with st.sidebar.expander("Quick Format Reference"):
                st.markdown("""
                * **Year:** `%Y` (2026) | `%y` (26)
                * **Month:** `%m` (06) | `%b` (Jun) | `%B` (June)
                * **Day:** `%d` (06)
                * **Time:** `%H` (Hours) | `%M` (Minutes) | `%S` (Seconds)
                """)
        else:
            self.chosen_format = self.DATE_FORMATS[selected_label]

    def _create_other_options_ux(self):
        st.sidebar.header("Other Options")
        self.show_info = st.sidebar.checkbox("Print data information", value=self.show_info)
        st.session_state["advanced_options"] = st.sidebar.checkbox("Advanced Options", value=False)

    def _html_extract_ux(self):
            save_numerical_figure = st.sidebar.checkbox("Save numerical figure into html.", value=False)
            save_categorical_figure = st.sidebar.checkbox("Save categorical figure into html.", value=False)
            save_heatmap_figure = st.sidebar.checkbox("Save heatmap figure into html.", value=False)
            self.save_numerical = save_numerical_figure
            self.save_categorical = save_categorical_figure
            self.save_heatmap = save_heatmap_figure

    def _html_download_ux(self):
        st.sidebar.download_button(
                label="Download EDA HTML Report",
                data=self._save_px_html(),
                file_name=f"eda_results_{self.dataset}.html",
                mime="text/html"
                )

    def _html_all_ux(self):
        st.sidebar.divider()
        st.sidebar.header("HTML extraction")
        self._html_extract_ux()
        self._html_download_ux()

    def _sidebar_ux(self):
        st.sidebar.divider()
        self._create_assignments_ux()
        st.sidebar.divider()
        self._create_numerical_ux()
        st.sidebar.divider()
        self._create_categorical_ux()
        st.sidebar.divider()
        self._create_other_options_ux()

    def __call__(self):
        self._df_initilazier()
        if isinstance(self.df, pd.DataFrame):
            self._create_saving_lists()
            self.numeric_cols, self.categorical_cols, self.all_cols = Data.column_assigner(data=self.df, date_column=self.date_column)
            st.title("SmartEda report")
            self._sidebar_ux()
            self._apply_transformations()
            self.numeric_cols, self.categorical_cols, self.all_cols = Data.column_assigner(data=self.df, date_column=self.date_column)
            self._create_render_plots()
            self._html_all_ux()
        else:
            st.title(f"Provide a dataframe in the sidebar to continue.")

if __name__ == "__main__":
    st.set_page_config("SmartEDA")
    smartedaui()()