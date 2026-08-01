import pandas as pd
import warnings
import io
import numpy as np

class Data:
    @staticmethod
    def load_data(df: str | pd.DataFrame | None = None, index_col: str | int | None = None, date_col: str | int | None = None, date_format="%d/%m/%Y",
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
                df.seek(0)
                df = pd.read_csv(df)
            
            elif isinstance(df, pd.DataFrame):
                df = df.copy()
        
            elif isinstance(df, str):
                df = pd.read_csv(df)
            else:
                if can_return_none:
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
            df = Data.handle_date_assignment(data=df, date_column=date_col, format=format)

        if return_columns:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=df, date_column=date_col)
            return df, numerical_columns, categorical_columns, all_cols
            
        return df

    @staticmethod
    def column_assigner(data:pd.DataFrame, numerical_columns:list = None,categorical_columns:list = None ,date_column=None) -> tuple[list, list, list]:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Provided dataframe is not a pd.DataFrame object.")

        if not isinstance(numerical_columns, list):
            numerical_columns = data.select_dtypes(include=np.number).columns.to_list()

        if not isinstance(categorical_columns, list):
            categorical_columns = data.select_dtypes(include=["str", "object", "category"]).columns.to_list()

        numerical_columns = [col for col in numerical_columns if col != date_column]
        categorical_columns = [col for col in categorical_columns if col != date_column]
        all_cols = numerical_columns + categorical_columns

        return numerical_columns, categorical_columns, all_cols
    
    @staticmethod
    def assign_date(data:pd.DataFrame, date_column, format="%Y-%m-%d") -> pd.DataFrame:
        try:
            data[date_column] = pd.to_datetime(data[date_column], format=format)
        except Exception as e:
            msg = f"Could not assign the column: {date_column} as date column for: {e} error. Continuing without assigning."
            warnings.warn(msg, category=UserWarning)
        return data

    @staticmethod
    def handle_date_assignment(data: pd.DataFrame, date_column, format="%Y-%m-%d") -> tuple[pd.DataFrame, list, list, list]:
        """Attempts to set date; drops column from column lists if it succeeds."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            data = Data.assign_date(data=data, date_column=date_column, format=format)

        warning_occurred = any(
            issubclass(w.category, UserWarning) and "Could not assign the column" in str(w.message)
            for w in captured
        )

        if not warning_occurred:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=data, date_column=date_column)


        return data, numerical_columns, categorical_columns, all_cols

    @staticmethod
    def assign_index(data: pd.DataFrame, index_column) -> pd.DataFrame:
        try:
            data.set_index(index_column, inplace=True)
        except Exception as e:
            msg =  f"Could not assign the column: {index_column} as index column for: {e} error. Continuing without assigning."
            warnings.warn(msg, category=UserWarning)
        return data

    @staticmethod
    def handle_index_assignment(data: pd.DataFrame, index_column, date_column=None) -> tuple[pd.DataFrame, list, list, list]:
        """Attempts to set index; drops column from columns lists if it succeeds."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            data = Data.assign_index(data=data, index_column=index_column)

        warning_occurred = any(
            issubclass(w.category, UserWarning) and "Could not assign the column" in str(w.message)
            for w in captured
        )

        if not warning_occurred:
            numerical_columns, categorical_columns, all_cols = Data.column_assigner(data=data, date_column=date_column)

        return data, numerical_columns, categorical_columns, all_cols

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

