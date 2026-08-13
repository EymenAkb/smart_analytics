from smart_analytics.core.data import Data
from smart_analytics.core.timeseries import SmartTimeSeries
import streamlit as st
from typing import Literal, get_args
import pandas as pd
import io
import warnings

marginal_literal = Literal[None, "box", "violin", "rug"]

class tsui(SmartTimeSeries):
    DATE_FORMATS = {
        "mixed (formats with different symbols)": "mixed",
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
    def __init__(self, df: pd.DataFrame | str | io.BytesIO | None = None, date_column = None, date_format="mixed",index_column = None, 
                 visualize_line_graph_numerical : bool = False, visualize_scatter :bool = False, visualize_area_graph:bool=False, visualize_line_categorical:bool=False,
                 visualize_treemap:bool=False,
                 save_scatter_graphs:bool=False, save_line_graphs_numerical:bool=False, save_area_graphs:bool=False, save_line_categorical:bool=False,
                 save_treemap:bool=False,
                 scatter_marginal_y:marginal_literal=None):
        
        self.df = Data.load_data(df=df, can_return_none=True, date_col=date_column, date_format=date_format)
        self.date_column = date_column
        self.date_format = date_format
        self.index_column = index_column
        self.visualize_line_numerical = visualize_line_graph_numerical
        self.visualize_scatter = visualize_scatter
        self.visualize_area = visualize_area_graph
        self.visualize_line_categorical = visualize_line_categorical
        self.visualize_treemap = visualize_treemap
        self.save_line_graphs_numerical = save_line_graphs_numerical
        self.save_scatter_graphs = save_scatter_graphs
        self.save_area_graphs = save_area_graphs
        self.save_line_categorical = save_line_categorical
        self.save_treemap = save_treemap
        
        if not scatter_marginal_y in get_args(marginal_literal):
            warnings.warn(f"{scatter_marginal_y} not in {get_args(marginal_literal)} falling back to None", category=UserWarning)
            self.marginal_y = None
        else:
            self.marginal_y = scatter_marginal_y

        if isinstance(self.df, pd.DataFrame):
            self._run()

    @st.cache_data
    def _load_dataframe(self, data_frame):
        return Data.load_data(df=data_frame, date_col=self.date_column, date_format=self.date_format)
    
    def _create_savings(self):
        self.figure_list = []
        self.figure_dict = {}
        self.numerical_line_figures = []
        self.numerical_line_figures_dict = {}
        self.scatter_figures = []
        self.scatter_figures_dict = {}
        self.area_figures = []
        self.area_figures_dict = {}
        self.categorical_line_figures = []
        self.categorical_line_figures_dict = {}
        self.treemap_figures = []
        self.treemap_figures_dict = {}

    def _run(self):
        self._create_savings()
        self._create_starter_ux()
        self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
        self._create_assignments_ux()
        self._apply_transformations()
        self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
        if Data.date_check(self.df, date_column=self.date_column):
            self._create_ux_sl()
            self._create_all_numerical_cols_sl()
            self._create_ux_al()
            self._create_all_categorical_ux_t()
        else:
            st.title("Provide date column from sidebar to start.")

    def _save_numerical_line(self, column, numerical_fig):
        self.figure_list.append(numerical_fig)
        self.figure_dict[column] = numerical_fig
        self.numerical_line_figures.append(numerical_fig)
        self.numerical_line_figures_dict[column] = numerical_fig                  

    def _save_scatter(self, column, scatter_figure):
        self.figure_list.append(scatter_figure)
        self.figure_dict[column] = scatter_figure
        self.scatter_figures.append(scatter_figure)
        self.scatter_figures_dict[column] = scatter_figure

    def _save_area(self, column, categorical_fig):
        self.figure_dict[column] = categorical_fig
        self.area_figures_dict[column] = categorical_fig
        self.figure_list.append(categorical_fig)
        self.area_figures.append(categorical_fig)

    def _save_categorical_line(self, column, categorical_fig):
        self.figure_list.append(categorical_fig)
        self.figure_dict[column] = categorical_fig
        self.categorical_line_figures.append(categorical_fig)
        self.categorical_line_figures_dict[column] = categorical_fig

    def _save_treemap(self, fig):
        self.figure_list.append(fig)
        self.figure_dict["treemap"] = fig
        self.treemap_figures.append(fig)
        self.treemap_figures_dict["treemap"] = fig

    def _create_ux_al(self):
        if not self.categorical_columns or not self.date_column or not (self.visualize_area or self.visualize_line_categorical):
            return
        st.title("Distribution visualizations - Categorical")

        for i,column in enumerate(self.categorical_columns):
            st.header(f"Distribution of {column}")
            c1, c2 = st.columns([1, 1])
            if self.visualize_line_categorical and self.visualize_area:
                with c1:
                    line_fig = self._create_line_visualization(column=column)
                    st.plotly_chart(line_fig, width="stretch")
                    if self.save_line_categorical:
                        self._save_categorical_line(column=column, categorical_fig=line_fig)
                with c2:
                    area_fig = self._create_area_visualization(column)
                    st.plotly_chart(area_fig, width="stretch")
                    if self.save_area_graphs:
                        self._save_area(column=column, categorical_fig=area_fig)

            else:
                if self.visualize_line_categorical:
                    fig = self._create_line_visualization(column=column)
                    st.plotly_chart(line_fig, width="stretch")
                    if self.save_line_categorical:
                        self._save_categorical_line(column=column, categorical_fig=fig)

                elif self.visualize_area:
                    fig = self._create_area_visualization(column)
                    st.plotly_chart(area_fig, width="stretch")
                    if self.save_area_graphs:
                        self._save_area(column=column, categorical_fig=fig)
                else:
                    return

                with c1:
                    st.plotly_chart(fig)

    def _create_all_categorical_ux_t(self):
        if self.visualize_treemap:
            st.header(f"Distrubition of all categorical columns in one graph")
            treemap_fig = self._create_treemap_visualization()
            st.plotly_chart(treemap_fig, width="stretch")
            if self.save_treemap:
                self._save_treemap(treemap_fig)

    def _create_ux_sl(self):
        if not self.numerical_columns or not self.date_column or not (self.visualize_scatter or self.visualize_line_graph):
            return

        st.title("Distribution visualizations - Numerical")

        for i, column in enumerate(self.numerical_columns):
            st.header(f"Distribution of {column}")
            c1, c2 = st.columns([1, 1])
            if self.visualize_scatter and self.visualize_line_graph:
                with c1:
                    ln_fig = self._create_line_visualization(column=column)
                    st.plotly_chart(ln_fig, width="stretch")
                    if self.save_line_graphs_numerical:
                        self._save_numerical_line(column=column, numerical_fig=ln_fig)
                with c2:
                    sc_fig = self._create_scatter_visualization(column=column)
                    st.plotly_chart(sc_fig, width="stretch")
                    if self.save_scatter_graphs:
                        self._save_scatter(column=column, scatter_figure=sc_fig)
            else:
                target = c1
                if self.visualize_line_graph:
                    fig = self._create_line_visualization(column=column)
                    if self.save_line_graphs_numerical:
                        self._save_line(column=column, numerical_fig=fig)
                elif self.visualize_scatter:
                    fig = self._create_scatter_visualization(column=column, marginal_y=self.scatter_marginal)
                    if self.save_scatter_graphs:
                        self._save_scatter(column=column, scatter_figure=fig)
                else:
                    return

                with target:
                    st.plotly_chart(fig, width="stretch")

    def _create_all_numerical_cols_sl(self):
        st.header(f"Distrubition of all columns on one graph")

        if self.visualize_line_graph:
            st.plotly_chart(self._create_line_visualization(column=self.numerical_columns), width="stretch")
        if self.visualize_scatter:
            st.plotly_chart(self._create_scatter_visualization(column=self.numerical_columns), width="stretch")

    def _create_assignments_ux(self):
        self.index_column = st.sidebar.selectbox("Index column", options= [None] + self.all_columns)
        self.date_column = st.sidebar.selectbox("Date column", options= [None] + self.all_columns)

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

    def _df_initilaizer(self):
        df = st.sidebar.file_uploader("Insert your DataFrame", type="csv")
        if df:
            self.df, self.numerical_columns, self.categorical_columns, self.all_columns = Data.load_data(df, return_columns=True)

    def _create_starter_ux(self):
        self.visualize_line_graph = st.sidebar.checkbox("Visualize Line Graph", value=False)
        self.visualize_scatter = st.sidebar.checkbox("Visualize Scatter Graph", value=False)
        self.scatter_marginal = st.sidebar.selectbox("Marginal y (advanced)", options=get_args(marginal_literal))

    def __call__(self):
        self._create_savings()
        if not self.df:
            self._df_initilaizer()
        self._create_starter_ux()
        if isinstance(self.df, pd.DataFrame):
            self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
            self._create_assignments_ux()
            self._apply_transformations()
            self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
            if Data.date_check(data=self.df, date_column=self.date_column):
                self._create_ux_sl()
                self._create_all_numerical_cols_sl()
                self._create_ux_al()
                self._create_all_categorical_ux_t()
            else:
                st.title("Provide date column from sidebar to start.")
        else:
            st.title("Provide a DataFrame from sidebar.")

if __name__ == "__main__":
    tsui(visualize_treemap=True, save_treemap=True)()