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
    def __init__(self):
        super().__init__()

    def _create_ux_sl(self):
        if not self.numerical_columns or not self.date_column or not (self.visualize_scatter or self.visualize_line_graph):
            return

        st.title("Distribution visualizations")

        c1, c2 = st.columns([1, 1])

        for i, column in enumerate(self.numerical_columns):
            st.header(f"Distribution of {column}")
            if self.visualize_scatter and self.visualize_line_graph:
                with c1:
                    st.plotly_chart(self._create_line_visualization(column=column), width="stretch")
                with c2:
                    st.plotly_chart(self._create_scatter_visualization(column=column, marginal_y=self.scatter_marginal), width="stretch")
            else:
                target = c1 if i%2==0 else c2
                if self.visualize_line_graph:
                    fig = self._create_line_visualization(column=column)
                elif self.visualize_scatter:
                    fig = self._create_scatter_visualization(column=column, marginal_y=self.scatter_marginal)
                else:
                    return

                with target:
                    st.plotly_chart(fig, width="stretch")


        st.header(f"Distrubition of all columns on one graph")
        if self.visualize_line_graph and self.visualize_scatter:
            with c1:
                st.plotly_chart(self._create_line_visualization(column=self.numerical_columns), width="stretch")
            with c2:
                st.plotly_chart(self._create_scatter_visualization(column=self.numerical_columns), width="stretch")
        else:
            target = c1 if target != c1 else c2
            if self.visualize_line_graph:
                with target:
                    st.plotly_chart(self._create_line_visualization(column=self.numerical_columns), width="stretch")
            elif self.visualize_scatter:
                with target:
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

    def _create_starter_ux(self):
        df = st.sidebar.file_uploader("Insert your DataFrame", type="csv")
        self.visualize_line_graph = st.sidebar.checkbox("Visualize Line Graph", value=False)
        self.visualize_scatter = st.sidebar.checkbox("Visualize Scatter Graph", value=False)
        self.scatter_marginal = st.sidebar.selectbox("Marginal y (advanced)", options=get_args(marginal_literal))

        if df:
            self.df, self.numerical_columns, self.categorical_columns, self.all_columns = Data.load_data(df, return_columns=True)

    def __call__(self):
        self._create_savings()
        self._create_starter_ux()
        if isinstance(self.df, pd.DataFrame):
            self._create_assignments_ux()
            self._apply_transformations()
            self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
            if self.date_column:
                self._create_ux_sl()
            else:
                st.title("Provide date column from sidebar to start.")
        else:
            st.title("Provide a DataFrame from sidebar.")

if __name__ == "__main__":
    tsui()()