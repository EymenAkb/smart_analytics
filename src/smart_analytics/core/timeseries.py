import pandas as pd
import numpy as np
import plotly.express as px
from typing import Literal, get_args
from smart_analytics.core.data import Data
from pandas.api.types import is_datetime64_any_dtype

marginal_literal = Literal[None, "box"]

class SmartTimeSeries:
    def __init__(self, df: pd.DataFrame | str | None = None, date_column = None, date_format="",index_column = None, 
                 visualize_line_graph : bool = False, visualize_scatter :bool = False, save_scatter_graphs:bool=False, save_line_graphs:bool=False):
        self.df = Data.load_data(df=df, can_return_none=True)
        self.date_column = date_column
        self.chosen_format = date_format
        self.index_column = index_column
        self.visualize_line = visualize_line_graph
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
        self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
        self._apply_transformations()
        if Data.date_check(self.df, date_column=self.date_column):
            self._render_plots()

    def _render_plots(self):
        if self.numerical_columns and Data.date_check(self.df, date_column=self.date_column):
            for column in self.numerical_columns:
                if self.visualize_line:
                    ln_fig = self._create_line_visualization(column=column)
                    if self.save_line_graphs:
                        self._save_line(column=column, numerical_fig=ln_fig)

                if self.visualize_scatter:
                    sc_fig = self._create_scatter_visualization(column=column)
                    if self.save_scatter_graphs:
                        self._save_scatter(column=column, scatter_figure=sc_fig)

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

    def _create_line_visualization(self, column):
        line_figure = px.line(self.df, x=self.date_column, y=column)
        return line_figure

    def _create_scatter_visualization(self, column, marginal_y: marginal_literal =None):
        scatter_figure = px.scatter(self.df, x=self.date_column, y=column, marginal_y=marginal_y)
        return scatter_figure

    def _apply_transformations(self):
        if self.index_column:
            self.df, self.numerical_columns, self.categorical_columns, self.all_columns = Data.handle_index_assignment(
                data=self.df, index_column=self.index_column, 
                numerical_columns=self.numerical_columns, 
                categorical_columns=self.categorical_columns,
                return_columns=True
            )
        
        if self.date_column:
            self.df, self.numerical_columns, self.categorical_columns, self.all_columns = Data.handle_date_assignment(
                data=self.df, date_column=self.date_column, 
                numerical_columns=self.numerical_columns, categorical_columns=self.categorical_columns, 
                format=self.chosen_format,
                return_columns=True
            )

    def __call__(self):
        if isinstance(self.df, pd.DataFrame) and Data.date_check(self.df, date_column=self.date_column):
            self._create_savings()
            self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
            self._apply_transformations()
            self._render_plots()

if __name__ == "__main__":
    pass