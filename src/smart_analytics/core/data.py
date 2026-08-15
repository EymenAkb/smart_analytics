import pandas as pd
import warnings
import io
from pathlib import Path
import numpy as np
from pandas.api.types import is_datetime64_any_dtype

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
                        categorical_columns: list | str | None = None ,date_column=None) -> tuple[list, list, list]:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Provided dataframe is not a pd.DataFrame object.")

        if isinstance(numerical_columns, str):
            numerical_columns = [numerical_columns]
        if isinstance(categorical_columns, str):
            categorical_columns = [categorical_columns]

        if not isinstance(numerical_columns, list):
            numerical_columns = data.select_dtypes(include=np.number).columns.to_list()

        if not isinstance(categorical_columns, list):
            categorical_columns = data.select_dtypes(include=["object", "category"]).columns.to_list()

        if not Data.date_check(data=data, date_column=date_column):
            date_column = None

        numerical_columns = [col for col in numerical_columns if col != date_column]
        categorical_columns = [col for col in categorical_columns if col != date_column]
        all_cols = numerical_columns + categorical_columns

        if date_column:
            if isinstance(date_column, str):
                all_cols.append(date_column)
            elif isinstance(date_column, (list, tuple, set)):
                all_cols.extend(date_column)
        
        return numerical_columns, categorical_columns, all_cols
    
    @staticmethod
    def assign_date(data: pd.DataFrame, date_column: list | str | int | None = None, numerical_columns: list | str | None=None, 
                    categorical_columns: list | str | None = None, date_format: list | str | dict ="mixed", 
                    return_columns: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
        if not isinstance(date_column, (str, int, list)):
            warnings.warn("Provided date column isn't in the waited formats returning the DataFrame without assigning.", category=UserWarning)
            return data
        
        try:
            if isinstance(date_column, list):
                for i,column in enumerate(date_column):
                    if isinstance(date_format, list):
                        fmt = date_format[i]
                    elif isinstance(date_format, dict):
                        fmt = date_format[column]
                    else:
                        fmt = date_format
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
    def return_iqr(data:pd.DataFrame, columns: str | list, return_columns: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"DataFrame isn't a pandas DataFrame.")
            
        if isinstance(columns, str):
            columns = [columns]
        
        if not isinstance(columns, (list, tuple)):
            return data
        
        data = data.copy()

        for col in columns:
            if col in data.columns:
                low, high = Data.calculate_iqr(data[col])
                data[col] = data[col].mask(
                    (data[col] < low) | (data[col] > high))
            else:
                warnings.warn(f"Column '{col}' not found in DataFrame.", category=UserWarning)
        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=data)
            return data, numerical_columns, categorical_columns, all_cols
        
        return data

    @staticmethod
    def date_check(data: pd.DataFrame, date_column: str | int | list = None) -> bool:
        if date_column is None:
            return False
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Provided DataFrame isn't a pandas DataFrame object.")
        if not isinstance(date_column, (str, int, list)):
            raise ValueError(f"The date column: {date_column} isn't in the awaited types: (str, int, list)")

        if isinstance(date_column, list):
            if len(date_column) == 1:
                date_column = date_column[0]
            else:
                if len(date_column) < 1:
                    raise ValueError("Provided date_column is empty.")
                for column in date_column:
                    val = is_datetime64_any_dtype(data[column])
                    if val == True:
                        continue
                    else:
                        return False
                return True
                
        if isinstance(date_column, int):
            date_column = data.columns[date_column]

        if isinstance(date_column, str):
            if date_column in data:
                return is_datetime64_any_dtype(data[date_column])
            else:
                return False

        else:
            raise TypeError(f"The date column: {date_column} isn't in the awaited types: (str, int, list)")