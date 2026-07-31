import pandas as pd
import numpy as np
import plotly.express as px
from typing import Literal, get_args
from smart_analytics.core.data import Data

marginal_literal = Literal[None, "box"]

class SmartTimeSeries:
    def __init__(self, df: pd.DataFrame | str | None = None, date_column = None, date_format="",index_column = None, 
                 visualize_line_graph : bool = False, visualize_scatter :bool = False):
        self.df = Data.load_data(df=df, can_return_none=True)
        self.date_column = date_column
        self.chosen_format = date_format
        self.index_column = index_column
        self.visualize_line = visualize_line_graph
        self.visualize_scatter = visualize_scatter
        self.line_figures = []
        self.line_figures_dict = {}
        self.scatter_figures = []
        self.scatter_figures_dict = {}
        if isinstance(self.df, pd.DataFrame):
            self._run()

    def _run(self):
        self._assign_columns()
        self._apply_transformations()
        self._render_plots()
    

    def _render_plots(self):
        if self.numerical_columns and self.date_column:
            for column in self.numerical_columns:
                if self.visualize_line:
                    fig = self._create_line_visualization(column=column)
                    self.line_figures.append(fig)
                    self.line_figures_dict[column] = fig

                if self.visualize_scatter:
                    fig = self._create_scatter_visualization(column=column)
                    self.scatter_figures.append(fig)
                    self.scatter_figures_dict[column] = fig

    def _create_line_visualization(self, column):
        line_figure = px.line(self.df, x=self.date_column, y=column)
        return line_figure

    def _create_scatter_visualization(self, column, marginal_y: marginal_literal =None):
        scatter_figure = px.scatter(self.df, x=self.date_column, y=column, marginal_y=marginal_y)
        return scatter_figure

    def _assign_columns(self):
        if isinstance(self.df, pd.DataFrame):
            self.numerical_columns = self.df.select_dtypes(include=np.number).columns.to_list()
            self.categorical_columns = self.df.select_dtypes(include=["object", "category", "str"]).columns.to_list()
        else:
            self.numerical_columns = []
            self.categorical_columns = []

        self.all_columns = self.numerical_columns + self.categorical_columns

    def _apply_transformations(self):
        if self.index_column:
            self.df, self.numerical_columns, self.categorical_columns, self.all_columns = Data.handle_index_assignment(
                data=self.df, index_column=self.index_column, 
                numerical_columns=self.numerical_columns, 
                categorical_columns=self.categorical_columns
            )
        
        if self.date_column:
            self.df, self.numerical_columns, self.categorical_columns, self.all_columns = Data.handle_date_assignment(
                data=self.df, date_column=self.date_column, 
                numerical_columns=self.numerical_columns, categorical_columns=self.categorical_columns, 
                format=self.chosen_format
            )

    def __call__(self):
        if isinstance(self.df, pd.DataFrame):
            self._assign_columns()
            self._apply_transformations()
            self._render_plots()



if __name__ == "__main__":
    df = pd.read_csv("Py.csv")
    eda = SmartTimeSeries(df=df, date_column="Date",date_format="%Y", visualize_line_graph=True, visualize_scatter=True)
    print(eda.line_figures)