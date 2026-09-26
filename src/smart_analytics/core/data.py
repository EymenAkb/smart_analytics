import pandas as pd
import warnings
import io
from pathlib import Path
import numpy as np
from pandas.api.types import is_datetime64_any_dtype
import os

class Data:
    @staticmethod
    def load_data(df: str | pd.DataFrame | io.BytesIO | Path | None = None, index_col: str | int | list | set | tuple | None = None, 
                  date_col: str | int | None = None, date_format="mixed", can_return_none: bool = False, 
                  return_columns: bool = False) -> None | pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
        """
        load_data method is a dataloading function, given the inputs it returns the cached dataframe.

        Parameters:
            df: [pd.DataFrame, str] = DataFrame object
            index_col: [str, int]
            date_col: [str, int]
            can_return_none: [bool]
        """
        try:
            if isinstance(df, io.BytesIO):
                df = pd.read_csv(df)
            
            elif isinstance(df, pd.DataFrame):
                df = df.copy()
        
            elif isinstance(df, (str, Path)):
                path = Path(df)
                ext = path.suffix.lower()

                if ".csv" in ext:
                    df = pd.read_csv(df)
                elif ".json" in ext:
                    df = pd.read_json(df)
                elif ".xlsx" in ext:
                    df = pd.read_excel(df)
                else:
                    if can_return_none:
                        warnings.warn("Error: Invalid or missing dataframe input occurred during DataFrame reading. Returning None.", category=UserWarning)
                        return None
                    raise ValueError("Invalid or missing dataframe input.")
            
            else:
                if can_return_none:
                    warnings.warn("Error: Invalid or missing dataframe input occurred during DataFrame reading. Returning None.", category=UserWarning)
                    return None
                raise ValueError("Invalid or missing dataframe input.")
            
        except Exception as e:
            if can_return_none:
                warnings.warn(f"Error: {e} occurred during DataFrame reading. Returning None.", category=UserWarning)
                return None
            else:
                raise ValueError(f"Couldn't read the dataframe: {e}")

        if index_col is not None:
            df = Data.assign_index(data=df, index_column=index_col)

        if date_col is not None:
            df = Data.assign_date(data=df, date_column=date_col, date_format=date_format)

        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=df, date_column=date_col)
            return df, numerical_columns, categorical_columns, all_cols
            
        return df

    @staticmethod
    def column_assigner(data:pd.DataFrame, numerical_columns: list | str | None = None, 
                        categorical_columns: list | str | None = None , date_column=None) -> tuple[list, list, list]:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Provided dataframe is not a pd.DataFrame object.")

        if not isinstance(numerical_columns, list):
            numerical_columns = data.select_dtypes(include=np.number).columns.to_list()

        if not isinstance(categorical_columns, list):
            categorical_columns = data.select_dtypes(include=["object", "category", pd.StringDtype()]).columns.to_list()

        numerical_columns = [col for col in numerical_columns if col != date_column]
        categorical_columns = [col for col in categorical_columns if col != date_column]

        if date_column is not None:
            try:
                Data.date_check(data=data, date_column=date_column)
            except ValueError:
                date_column = None

            if not isinstance(date_column, list):
                date_column = [date_column]
            all_cols = numerical_columns + categorical_columns + date_column
        else:
            all_cols = numerical_columns + categorical_columns
        
        return numerical_columns, categorical_columns, all_cols
    
    @staticmethod
    def assign_date(data: pd.DataFrame, date_column: list | str | int | None = None, numerical_columns: list | str | None=None, 
                    categorical_columns: list | str | None = None, date_format: str | list | dict | tuple |set ="mixed", 
                    return_columns: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
        if not isinstance(date_column, (str, int, list, tuple, set)):
            warnings.warn("Provided date column isn't in the waited formats returning the DataFrame without assigning.", category=UserWarning)
            return data
        
        try:
            if isinstance(date_format, set):
                date_format = list(set)
            if isinstance(date_column, (list, set, tuple)):
                for i,column in enumerate(date_column):
                    if isinstance(date_format, (list, tuple)):
                        fmt = date_format[i]
                    elif isinstance(date_format, dict):
                        fmt = date_format[column]
                    elif isinstance(date_format, str):
                        fmt = date_format
                    else:
                        fmt = "mixed"
                    data[column] = pd.to_datetime(data[column], format=fmt)

            if isinstance(date_column, int):
                date_column = data.columns[date_column]

            if isinstance(date_column, str):
                data[date_column] = pd.to_datetime(data[date_column], format=date_format)

        except Exception as e:
            msg = f"Could not assign the column: {date_column} as date column for: {e} error. Continuing without assigning."
            warnings.warn(msg, category=UserWarning)

        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=data, numerical_columns=numerical_columns, 
                                                                categorical_columns=categorical_columns, date_column=date_column)
            return data, numerical_columns, categorical_columns, all_cols
        
        return data

    @staticmethod
    def assign_index(data:pd.DataFrame, index_column: str | int | list | set | tuple | None = None, numerical_columns: list | str | None = None, 
                    categorical_columns: list | str | None = None, date_column: list | str | int | None = None, 
                    return_columns: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
        if not isinstance(index_column, (str, int, list, tuple, set)):
            warnings.warn("index column isn't in the waited formats returning the DataFrame normally", category=UserWarning)
            return data
        
        try:
            if isinstance(index_column, list):
                data.set_index(index_column, inplace=True)

            elif isinstance(index_column, set):
                data.set_index(sorted(index_column), inplace=True)

            elif isinstance(index_column, tuple):
                data.set_index(list(index_column), inplace=True)

            elif isinstance(index_column, int):
                index_column = data.columns[index_column]
                data.set_index(index_column, inplace=True)

            elif isinstance(index_column, str):
                data.set_index(index_column, inplace=True)

        except Exception as e:
            msg =  f"Could not assign the column: {index_column} as index column for: {e} error. Continuing without assigning."
            warnings.warn(msg, category=UserWarning)

        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=data, numerical_columns=numerical_columns, 
                                                                    categorical_columns=categorical_columns, date_column=date_column)
            return data, numerical_columns, categorical_columns, all_cols

        return data

    @staticmethod
    def calculate_iqr(column:pd.Series):
        q_25 = column.quantile(0.25)
        q_75 = column.quantile(0.75)
        iqr = q_75 - q_25
        low = q_25 - (1.5 * iqr)
        high = q_75 + (1.5 * iqr)
        return (low, high)

    @staticmethod
    def return_iqr(data:pd.DataFrame, columns: str| int | list[str | int] | tuple[str | int] | set[str | int], 
                   return_columns: bool = False, copy:bool=True) -> pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"DataFrame isn't a pandas DataFrame.")
            
        if isinstance(columns, str):
            columns = [columns]

        if isinstance(columns, int):
            columns = [data.columns[columns]]
        
        if not isinstance(columns, (list, tuple, set)):
            raise TypeError("Provided columns aren't in the awaited types: [str, int, list, tuple, set]")
        
        data = data.copy() if copy else data

        for i, col in enumerate(columns):
            if isinstance(col, int):
                if 0 <= col < data.shape[1]:
                    columns[i] = data.columns[col]
                else:
                    raise IndexError(f"Column index {col} is out of bounds for DataFrame with {data.shape[1]} columns.")
            elif not isinstance(col, str):
                raise TypeError(f"Provided column {col!r}, aren't in the awaited types: [str, int].")

        for col in columns:
            if col in data.columns:
                low, high = Data.calculate_iqr(data[col])
                data[col] = data[col].mask(
                    (data[col] < low) | (data[col] > high))
            else:
                raise ValueError(f"Provided column {col!r} isn't in {data.columns}")
        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=data)
            return data, numerical_columns, categorical_columns, all_cols
        
        return data

    @staticmethod
    def date_check(data: pd.DataFrame, date_column: str | int | list[str | int] | None = None) -> bool:
        if not isinstance(data, pd.DataFrame) or date_column is None:
            return False

        if isinstance(date_column, (str, int)) and not isinstance(date_column, bool):
            columns = [date_column]

        elif isinstance(date_column, (list, tuple, set)) and date_column:
            columns = date_column

        else:
            return False

        for col in columns:
            if isinstance(col, bool):
                return False

            if isinstance(col, int):
                if col < 0 or col >= data.shape[1]:
                    return False
                series = data.iloc[:, col]


            elif isinstance(col, str):
                if col not in data.columns:
                    return False

                series = data[col]

                if isinstance(series, pd.DataFrame):
                    return False

            else:
                return False

            if not is_datetime64_any_dtype(series):
                return False
            
        return True