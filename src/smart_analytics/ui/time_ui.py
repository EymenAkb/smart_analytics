from smart_analytics.core.data import Data
from smart_analytics.core.timeseries import SmartTimeSeries
import streamlit as st
from typing import Literal, get_args
import pandas as pd

marginal_literal = Literal[None, "box"]

class tsui(SmartTimeSeries):
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
    def __init__(self, df: pd.DataFrame | str | None = None, date_column = None, date_format="",index_column = None, 
                 visualize_line_graph : bool = False, visualize_scatter :bool = False, save_scatter_graphs:bool=False, save_line_graphs:bool=False):
        self.df = Data.load_data(df=df, can_return_none=True)
        self.date_column = date_column
        self.chosen_format = date_format
        self.index_column = index_column
        self.visualize_line_graph = visualize_line_graph
        self.visualize_scatter = visualize_scatter
        self.save_line_graphs = save_line_graphs
        self.save_scatter_graphs = save_scatter_graphs

        if isinstance(self.df, pd.DataFrame):
            self._run()

    def _create_savings(self):
        self.figure_list = []
        self.figure_dict = {}
        self.line_figures = []
        self.line_figures_dict = {}
        self.scatter_figures = []
        self.scatter_figures_dict = {}

    def _run(self):
        self._create_savings()
        self._create_starter_ux()
        self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
        self._create_assignments_ux()
        self._apply_transformations()
        self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
        if Data.date_check(self.df, date_column=self.date_column):
            self._create_ux_sl()
            self._create_all_cols_sl()
        else:
            st.title("Provide date column from sidebar to start.")

    def _save_scatter(self, column, scatter_figure):
        self.figure_list.append(scatter_figure)
        self.figure_dict[column] = scatter_figure
        self.scatter_figures.append(scatter_figure)
        self.scatter_figures_dict[column] = scatter_figure

    def _save_line(self, column, numerical_fig):
        self.figure_list.append(numerical_fig)
        self.figure_dict[column] = numerical_fig
        self.line_figures.append(numerical_fig)
        self.line_figures_dict[column] = numerical_fig

    def _create_ux_sl(self):
        if not self.numerical_columns or not self.date_column or not (self.visualize_scatter or self.visualize_line_graph):
            return

        st.title("Distribution visualizations")

        for i, column in enumerate(self.numerical_columns):
            st.header(f"Distribution of {column}")
            c1, c2 = st.columns([1, 1])
            if self.visualize_scatter and self.visualize_line_graph:
                with c1:
                    ln_fig = self._create_line_visualization(column=column)
                    st.plotly_chart(ln_fig, width="stretch")
                    if self.save_line_graphs:
                        self._save_line(column=column, numerical_fig=ln_fig)
                with c2:
                    sc_fig = self._create_scatter_visualization(column=column, marginal_y=self.scatter_marginal)
                    st.plotly_chart(sc_fig, width="stretch")
                    if self.save_scatter_graphs:
                        self._save_scatter(column=column, scatter_figure=sc_fig)
            else:
                target = c1
                if self.visualize_line_graph:
                    fig = self._create_line_visualization(column=column)
                    if self.save_line_graphs:
                        self._save_line(column=column, numerical_fig=fig)
                elif self.visualize_scatter:
                    fig = self._create_scatter_visualization(column=column, marginal_y=self.scatter_marginal)
                    if self.save_scatter_graphs:
                        self._save_scatter(column=column, scatter_figure=fig)
                else:
                    return

                with target:
                    st.plotly_chart(fig, width="stretch")

    def _create_all_cols_sl(self):
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
                self._create_all_cols_sl()
            else:
                st.title("Provide date column from sidebar to start.")
        else:
            st.title("Provide a DataFrame from sidebar.")

if __name__ == "__main__":
    pass