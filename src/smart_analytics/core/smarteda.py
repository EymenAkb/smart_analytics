import pandas as pd
import numpy as np
import os
import plotly.express as px
from typing import Literal, get_args
import io
import warnings
from smart_analytics.core.data import Data

IQRHandleMethod = Literal["ignore", "nan"]
HistogramMarginalValues = Literal[None, "box", "rug", "violin"]
HistogramBarModes = Literal["relative", "overlay", "group", "stack"]

class SmartEDA:
    """
    SmartEDA provides automated exploratory data analysis (EDA)
    including data intuiton and visualization for Data Analytics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    
    visualize_numerical : bool, default=False
        Whether to display numeric feature distributions.
    
    visualize_categorical : bool, default=False
        Whether to display categorical feature distributions.
    
    visualize heatmap : bool, deafult=False
        Wheter to display correlation matrix.

    save_numerical_figures : bool, default=False
        Wheter to save the numerical visualizations.
    
    histogram_marginal: [None, "box", "rug", "violin"], default=None
        What to show on marginal value over Histograms. (If color is set to anything marginal might fall back to "rug")
    
    histogram_color_method : bool, default=False
        Wheter to use column as color options or not for histograms.

    histogram_bar_mode: ["relative", "overlay", "group", "stack"], default="relative"
        What to pass into histograms barmode parameter.
    
    handle_iqr: ["ignore", "nan"], default="ignore"
        How to handle iqr values. (outliers)
        
    show_iqr_box: bool, default=False
        Wheter to display box plots for iqr.

    save_categorical_figures: bool, default=False
        Wheter to save the categorical visualizations.

    save_heatmap_figure: bool, default=False
        Wheter to save the correlation matrix visualization.

    show_info: bool, default=False
        Wheter to display information about dataset.

    dataset_name: str, default="dataset"
        What to display as dataset name for summurazing purposes.

    date_column: str | int | None, default=None
        Assigner for Data's date column.
    
    date_format: str, default="mixed"
        Date format for the provided date column.

    index_column: str | int | list| tuple | set | None, default=None
        Assigner for provided dataframes index column(s).
    """
    def __init__(self,
        df: pd.DataFrame | str | io.BytesIO = None,
        visualize_numerical: bool = False,
        visualize_categorical: bool = False,
        visualize_heatmap: bool = False,
        save_numerical_figures: bool = False,
        histogram_marginal: HistogramMarginalValues = None,
        histogram_color_method: bool = False,
        histogram_bar_mode: HistogramBarModes = "relative",
        handle_iqr: IQRHandleMethod = "ignore",
        show_iqr_box: bool = False,
        save_categorical_figures: bool = False,
        save_heatmap_figure: bool = False,
        show_info: bool = False,
        dataset_name:str = "dataset",
        date_column: str | int | None = None,
        date_format: str = "mixed",
        index_column: str | int | list| tuple | set | None = None):
        
        self.iqr_methods = get_args(IQRHandleMethod)
        self.hist_marginal_methods = get_args(HistogramMarginalValues)
        self.hist_bar_modes = get_args(HistogramBarModes)
        handle_iqr = handle_iqr.lower().strip()

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
                self._create_histogram_visualization(column=column, marginal=self.histogram_marginal, 
                                                     barmode=self.histogram_bar_mode, color=self.histogram_color_method)

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

    def _create_histogram_visualization(self, column, marginal=None, barmode="relative", color=False):
        if color:
            color = column
        else:
            color = None
        numeric_figure = px.histogram(self.df, x=column, marginal=marginal, barmode=barmode, color=color)
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
        return self.df.columns[idx]

    def __len__(self):
        return len(self.df.columns)
        

if __name__ == "__main__":
    pass