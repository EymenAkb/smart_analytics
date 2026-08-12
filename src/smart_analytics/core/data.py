import pandas as pd
import warnings
import io
import numpy as np
from pandas.api.types import is_datetime64_any_dtype

class Data:
    @staticmethod
    def load_data(df: str | pd.DataFrame | None = None, index_col: str | int | None = None, date_col: str | int | None = None, date_format="mixed",
                  can_return_none: bool = False, return_columns:bool=False) -> None | pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
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
        
            elif isinstance(df, str):
                df = pd.read_csv(df)
            
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

        if index_col:
            df = Data.handle_index_assignment(data=df, index_column=index_col)

        if date_col:
            df = Data.handle_date_assignment(data=df, date_column=date_col, date_format=date_format)

        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=df, date_column=date_col)
            return df, numerical_columns, categorical_columns, all_cols
            
        return df

    @staticmethod
    def column_assigner(data:pd.DataFrame, numerical_columns:list = None,categorical_columns:list = None ,date_column=None) -> tuple[list, list, list]:
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
            elif isinstance(date_column, list):
                all_cols + date_column
        
        return numerical_columns, categorical_columns, all_cols
    
    @staticmethod
    def assign_date(data:pd.DataFrame, date_column, date_format="mixed") -> pd.DataFrame:
        if not isinstance(date_column, (int, str)):
            warnings.warn("Provided date column isn't in the waited formats returning the DataFrame without assigning.", category=UserWarning)
            return data
        try:
            data[date_column] = pd.to_datetime(data[date_column], format=date_format)
        except Exception as e:
            msg = f"Could not assign the column: {date_column} as date column for: {e} error. Continuing without assigning."
            warnings.warn(msg, category=UserWarning)
        return data

    @staticmethod
    def handle_date_assignment(data: pd.DataFrame, date_column=None, numerical_columns:list=None, categorical_columns:list = None, 
                               date_format="mixed", return_columns:bool=False) -> pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
        """Attempts to set date; drops column from column lists if it succeeds."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            data = Data.assign_date(data=data, date_column=date_column, date_format=date_format)

        warning_occurred = any(
            issubclass(w.category, UserWarning) and "Could not assign the column" in str(w.message)
            for w in captured
        )

        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=data, numerical_columns=numerical_columns, 
                                                                categorical_columns=categorical_columns, date_column=date_column)
            return data, numerical_columns, categorical_columns, all_cols
        
        return data

    @staticmethod
    def assign_index(data: pd.DataFrame, index_column=None) -> pd.DataFrame:
        if not isinstance(index_column, (str, int)):
            warnings.warn("index column isn't in the waited formats returning the DataFrame normally", category=UserWarning)
            return data
        try:
            if isinstance(index_column, int):
                index_column = data.columns[index_column]
            data.set_index(index_column, inplace=True)
        except Exception as e:
            msg =  f"Could not assign the column: {index_column} as index column for: {e} error. Continuing without assigning."
            warnings.warn(msg, category=UserWarning)
        return data

    @staticmethod
    def handle_index_assignment(data:pd.DataFrame, index_column=None, numerical_columns=None, 
                                categorical_columns=None, date_column=None, return_columns:bool=False) -> pd.DataFrame | tuple[pd.DataFrame, list, list, list]:
        """Attempts to set index; drops column from columns lists if it succeeds."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            data = Data.assign_index(data=data, index_column=index_column)

        warning_occurred = any(
            issubclass(w.category, UserWarning) and "Could not assign the column" in str(w.message)
            for w in captured
        )

        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=data, numerical_columns=numerical_columns, 
                                                                    categorical_columns=categorical_columns, date_column=date_column)
            return data, numerical_columns, categorical_columns, all_cols

        return data

    @staticmethod
    def calculate_iqr(column):
        q_25 = column.quantile(0.25)
        q_75 = column.quantile(0.75)
        iqr = q_75 - q_25
        low = q_25 - (1.5 * iqr)
        high = q_75 + (1.5 * iqr)
        return (low, high)


    @staticmethod
    def return_iqr(data, columns):
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
        return data

    @staticmethod
    def date_check(data: pd.DataFrame, date_column: str | int = None) -> bool:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Provided DataFrame isn't a pandas DataFrame object.")

        if date_column is None or date_column not in data.columns:
            return False
        
        return is_datetime64_any_dtype(data[date_column])