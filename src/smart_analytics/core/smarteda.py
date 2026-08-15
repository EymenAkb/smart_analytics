import pandas as pd
import numpy as np
import os
import plotly.express as px
from typing import Literal, get_args
import io
import warnings
from smart_analytics.core.data import Data

IQRHandleMethod = Literal["ignore", "nan"]


class SmartEDA:
    """
    SmartEDA provides automated exploratory data analysis (EDA)
    including data intuiton and visualization for pandas DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    
    visualize_numerical : bool, default=False
        Whether to display numeric feature distributions.
    
    visualize_categorical : bool, default=False
        Whether to display categorical feature distributions.
    """
    def __init__(self,
        df: pd.DataFrame | str | io.BytesIO = None,
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
        date_format: str = None,
        index_column: str | int | None = None):
        
        self.iqr_methods = get_args(IQRHandleMethod)
        handle_iqr = handle_iqr.lower().strip()

        if not handle_iqr in self.iqr_methods:
            warnings.warn(f"Invalid method '{handle_iqr}'. Expected one of {self.iqr_methods}, falling back to 'ignore'.", category=UserWarning)
            handle_iqr = "ignore"

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
        if saving_directory:
            self.save_path = os.path.join(saving_directory)
        else:
            self.save_path = os.path.join(".", dataset_name)

        self.df = Data.load_data(df=df, index_col=index_column, date_col=date_column, date_format=date_format, can_return_none=True, return_columns=False)

        if isinstance(self.df, pd.DataFrame):
            self._run()

    def _create_saving_lists(self):
        self.numerical_hist_list = []
        self.numerical_hist_dict = {}
        self.numerical_box_list = []
        self.numerical_box_dict = {}
        self.categorical_bar_list = []
        self.categorical_bar_dict = {}
        self.heatmap_list = []
        self.heatmap_dict = {}
        self.figure_list = []
        self.figure_dict = {}

    def _run(self):
        self._create_saving_lists()
        self.numeric_cols, self.categorical_cols, self.all_cols = Data.column_assigner(data=self.df, date_column=self.date_column)
        self._apply_transformations()
        self._create_render_plots()

    def _create_render_plots(self):
        if self.show_info:
            self.info_show()

        if (self.visnum or self.save_numerical) and isinstance(self.numeric_cols, list):
            for column in self.numeric_cols:
                self._create_histogram_visualization(column=column)

        if (self.viscat or self.save_categorical) and isinstance(self.categorical_cols, list):
            for column in self.categorical_cols:
                self._create_bar_figure(column=column)

        if (self.vishm or self.save_heatmap) and isinstance(self.numeric_cols, list):
            self._create_heatmap_vis()

        if (self.save_categorical or self.save_heatmap or self.save_numerical) and isinstance(self.numeric_cols, list) and isinstance(self.categorical_cols, list):
            self._save_px_html()

    def info_show(self):
        print("Information")
        print(self.df.info())

        print("\n5 sample from dataset:")
        print(self.df.head())

        print("\nEmpty rows per column:")
        print(self.df.isnull().sum())

    def _create_histogram_visualization(self, column):
        numeric_figure = px.histogram(self.df, x=column, color=column, barmode="group", marginal="box")
        numeric_figure.update_layout(bargap=0.0)

        if self.save_numerical:
            self.figure_list.append(numeric_figure)
            self.figure_dict[column] = numeric_figure

            self.numerical_hist_list.append(numeric_figure)
            self.numerical_hist_dict[column] = numeric_figure

        return numeric_figure

    def _create_box_plot(self, column):
        box_figure = px.box(self.df, x=column)

        if self.save_numerical and self.show_iqr_box:
            self.figure_list.append(box_figure)
            self.figure_dict[column] = box_figure

            self.numerical_box_list.append(box_figure)
            self.numerical_box_dict[column] = box_figure

        return box_figure

    def _create_bar_figure(self, column):
        categorical_figure = px.bar(self.df, column)

        if self.save_categorical:
            self.figure_list.append(categorical_figure)
            self.figure_dict[column] = categorical_figure

            self.categorical_bar_dict[column] = categorical_figure
            self.categorical_bar_list.append(categorical_figure)

        return categorical_figure

    def _create_heatmap_vis(self):

        corr_mat = self.df[self.numeric_cols].corr()
        hm_fig = px.imshow(corr_mat, text_auto=".2f", color_continuous_scale="RdBu_r")

        if self.save_heatmap:
            self.figure_list.append(hm_fig)
            self.figure_dict["hetmap"] = hm_fig

            self.heatmap_dict["heatmap"] = hm_fig
            self.heatmap_list.append(hm_fig)

        return hm_fig

    def _apply_transformations(self):
        if self.index_column:
            self.df, self.numeric_cols, self.categorical_cols, self.all_cols = Data.assign_index(
                data=self.df, index_column=self.index_column, 
                numerical_columns=self.numeric_cols, 
                categorical_columns=self.categorical_cols,
                date_column=self.date_column,
                return_columns=True
            )
        
        if self.date_column:
            self.df, self.numeric_cols, self.categorical_cols, self.all_cols = Data.assign_date(
                data=self.df, date_column=self.date_column, 
                numerical_columns=self.numeric_cols, categorical_columns=self.categorical_cols, 
                date_format=self.date_format, 
                return_columns=True
            )

        if self.handle_iqr == "nan":
            self.df = Data.return_iqr(self.df, columns=self.numeric_cols)

    def __call__(self, df: pd.DataFrame | str | io.BytesIO = None):
        read_df = Data.load_data(df, can_return_none=True)
        if read_df is not None:
            self.df = read_df

        self._create_saving_lists()

        if isinstance(self.df, pd.DataFrame):
            self.numeric_cols, self.categorical_cols, self.all_cols = Data.column_assigner(data=self.df, date_column=self.date_column)
            self._apply_transformations()
            self._create_render_plots()

    def _save_px_html(self):
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '    <meta charset="utf-8">',
            '    <title>EDA Results</title>',
            '    <script src="https://cdn.plot.ly/plotly-3.3.0.min.js"></script>',
            "</head>",
            "<body>",
            f"    <h1>EDA Results - {self.dataset}</h1>"
        ]
        
        for graph in self.figure_list:
            graph_html = graph.to_html(full_html=False, include_plotlyjs=False)
            html_content.append(graph_html)
            html_content.append("<br><hr><br>\n")
            
        html_content.extend(["</body>", "</html>"])
        self.html_fig = "\n".join(html_content)
        return self.html_fig


    def __str__(self):
        result = f"""
Dataset details:

Dataset name: {self.dataset}
Total column numbers: {len(self.numeric_cols + self.categorical_cols)}

----------------

Empty rows per column:
{self.df.isnull().sum()}

----------------

Dataset information:
{self.df.info()}

----------------

Some samples from the dataset:
{self.df.head()}
"""
        return result
    
    def __getitem__(self, idx):
        return self.df[self.all_cols[idx]]
        

if __name__ == "__main__":
    import seaborn as sns
    df = sns.load_dataset("titanic")
    eda = SmartEDA(df=df)

    #handle_date_assignment