from smart_analytics.core.smarteda import SmartEDA
from smart_analytics.core.data import Data
import streamlit as st
from typing import Literal, get_args
import pandas as pd
import io

IQRHandleMethod = Literal["ignore", "nan"]

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
    def __init__(self, df: pd.DataFrame | str | io.BytesIO = None,
        visualize_numerical: bool = False,
        visualize_categorical: bool = False,
        visualize_heatmap: bool = False,
        save_numerical_figures: bool = False,
        save_categorical_figures: bool = False,
        save_heatmap_figure: bool = False,
        show_info: bool = False,
        dataset_name:str = "dataset",
        saving_directory: str | None = None,
        handle_iqr: IQRHandleMethod = "ignore",
        show_iqr_box: bool = False,
        date_column: str | int | None = None,
        index_column: str | int | None = None):
        super().__init__(df, visualize_numerical, visualize_categorical, visualize_heatmap, save_numerical_figures,save_categorical_figures,
                        save_heatmap_figure, save_heatmap_figure, show_info, dataset_name, saving_directory, handle_iqr, show_iqr_box,
                        date_column, index_column)

    @st.cache_data
    def _load_cached_data(uploaded_file):
        return Data.load_data(uploaded_file, can_return_none=True)

    def _create_numerical_visualization(self):
        if not self.numeric_cols:
            return
        st.header("Numerical visualizations")
        c1, c2 = st.columns([2, 1] if self.show_iqr_box else [1, 1])

        for i, column in enumerate(self.numeric_cols):
            if self.show_iqr_box:
                with c1:
                    st.plotly_chart(self._create_histogram_visualization(column=column), width="stretch")
                with c2:
                    st.plotly_chart(self._create_box_plot(column=column), width="stretch")
            else:
                target = c1 if i % 2 == 0 else c2
                with target:
                    st.plotly_chart(self._create_histogram_visualization(column=column), width="stretch")

    def _create_categorical_vis(self):
        if not self.categorical_cols:
            return
        
        st.header("Categorical visualizations")
        c1, c2 = st.columns([1,1])

        for i, column in enumerate(self.categorical_cols):
            target = c1 if not i % 2 else c2
            with target:
                st.plotly_chart(self._create_bar_plot(column=column), width="stretch")

    def _create_heatmap_figure(self):
        if not self.numeric_cols:
            return
        
        st.header("Correlation heatmap")
        st.plotly_chart(self._create_heatmap_vis(), width="stretch")

    def _create_starter_ux(self):
        df = st.sidebar.file_uploader("Upload csv file", type=["csv"])
        self.visnum = st.sidebar.checkbox("Visualize numerical", value=self.visnum)
        self.show_iqr_box = st.sidebar.checkbox("Show box graph", value=self.show_iqr_box,
            help="If handle iqr is set to 'nan' the box plots might show wrong details!")
        self.viscat = st.sidebar.checkbox("Visualize categorical", value=self.viscat)
        self.vishm = st.sidebar.checkbox("Visualize heatmap", value=self.vishm)
        self.show_info = st.sidebar.checkbox("Print data information", value=self.show_info)
        self.handle_iqr = st.sidebar.selectbox("Handle iqr method", get_args(IQRHandleMethod),
            help="If handle iqr is set to 'nan' the box plots might show wrong details!")
        
        if df:
            self.df = self._load_cached_data(df)

    def _create_assignments_ux(self):
        
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

    def _html_extract_ux(self):
            st.sidebar.title("HTML extraction")
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

    def __call__(self):
        self._create_starter_ux()
        if isinstance(self.df, pd.DataFrame):
            self._assign_column_categories()
            st.title("SmartEda report")
            st.sidebar.divider()
            self._create_assignments_ux()
            self._apply_transformations()
            self._create_render_plots()
            self._html_extract_ux()
            self._html_download_ux()
        else:
            st.title(f"Provide a dataframe in the sidebar to continue.")

if __name__ == "__main__":
    st.set_page_config("SmartEDA")
    smartedaui()()