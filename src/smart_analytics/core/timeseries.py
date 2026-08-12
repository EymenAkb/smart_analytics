import pandas as pd
import numpy as np
import plotly.express as px
from typing import Literal, get_args
from smart_analytics.core.data import Data
import io

marginal_literal = Literal[None, "box"]

class SmartTimeSeries:
    def __init__(self, df: pd.DataFrame | str | io.BytesIO | None = None, date_column = None, date_format="mixed",index_column = None, 
                 visualize_line_graph_numericak : bool = False, visualize_scatter :bool = False, visualize_area_graph:bool=False, visualize_line_categorical:bool=False,
                 save_scatter_graphs:bool=False, save_line_graphs_numerical:bool=False, save_area_graphs:bool=False, save_line_categorical:bool=False):
        self.df = Data.load_data(df=df, can_return_none=True, date_col=date_column, date_format=date_format)
        self.date_column = date_column
        self.date_format = date_format
        self.index_column = index_column
        self.visualize_line_numerical = visualize_line_graph_numericak
        self.visualize_scatter = visualize_scatter
        self.visualize_area = visualize_area_graph
        self.visualize_line_categorical = visualize_line_categorical
        self.save_line_graphs_numerical = save_line_graphs_numerical
        self.save_scatter_graphs = save_scatter_graphs
        self.save_area_graphs = save_area_graphs
        self.save_line_categorical = save_line_categorical

        if isinstance(self.df, pd.DataFrame):
            self._run()

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

    def _run(self):
        self._create_savings()
        self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
        self._apply_transformations()
        if Data.date_check(self.df, date_column=self.date_column):
            self._render_plots()

    def _render_plots(self):
        if self.numerical_columns:
            for column in self.numerical_columns:
                if self.visualize_line_numerical:
                    ln_fig = self._create_line_visualization(column=column)
                    if self.save_line_graphs_numerical:
                        self._save_numerical_line(column=column, numerical_fig=ln_fig)

                if self.visualize_scatter:
                    sc_fig = self._create_scatter_visualization(column=column)
                    if self.save_scatter_graphs:
                        self._save_scatter(column=column, scatter_figure=sc_fig)

        if self.categorical_columns:
            if self.visualize_area:
                area_fig = self._create_area_visualization()
                if self.save_area_graphs:
                    self._save_area(categorical_fig=area_fig)
            for column in self.categorical_columns:
                if self.visualize_line_categorical:
                    line_fig = self._create_line_visualization(column=column)
                    if self.save_line_categorical:
                        self._save_categorical_line(column=column, categorical_fig=line_fig)
                        
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

    def _save_area(self, categorical_fig):
        self.figure_list.append(categorical_fig)
        self.figure_dict["area"] = categorical_fig
        self.area_figures.append(categorical_fig)
        self.area_figures_dict["area"] = categorical_fig

    def _save_categorical_line(self, column, categorical_fig):
        self.figure_list.append(categorical_fig)
        self.figure_dict[column] = categorical_fig
        self.categorical_line_figures.append(categorical_fig)
        self.categorical_line_figures_dict[column] = categorical_fig

    def _create_line_visualization(self, column):
        line_figure = px.line(self.df, x=self.date_column, y=column)
        return line_figure

    def _create_scatter_visualization(self, column, marginal_y: marginal_literal =None):
        scatter_figure = px.scatter(self.df, x=self.date_column, y=column, marginal_y=marginal_y)
        return scatter_figure

    def _create_area_visualization(self):
        area_figure = px.area(self.df, x=self.date_column, y=self.categorical_columns,title=f"Distribution of all categorical values", markers=True)
        return area_figure

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
                date_format=self.date_format,
                return_columns=True
            )

    def __call__(self, df: pd.DataFrame | str | io.BytesIO | None, date_column:str=None, date_format:str="mixed"):
        if not isinstance(self.df, pd.DataFrame):
            self.df = Data.load_data(df=df, date_col=date_column, date_format=date_format)
        if isinstance(self.df, pd.DataFrame) and Data.date_check(self.df, date_column=self.date_column):
            self._create_savings()
            self.numerical_columns, self.categorical_columns, self.all_columns = Data.column_assigner(data=self.df, date_column=self.date_column)
            self._apply_transformations()
            self._render_plots()

if __name__ == "__main__":
    df = pd.read_csv("Py.csv")
    ts = SmartTimeSeries(df=df, date_column="Date", visualize_line_categorical=True, save_line_categorical=True, date_format="%Y")