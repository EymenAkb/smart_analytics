import pandas as pd
import warnings
import streamlit as st
import io

class Data:
    @staticmethod
    def load_data(df: str | pd.DataFrame | None = None, index_col: str | int | None = None, date_col: str | int | None = None, date_format="%d/%m/%Y",can_return_none: bool = False) -> None | pd.DataFrame:
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
                st.error(f"Couldn't read the dataframe: {e}")
                raise ValueError(f"Couldn't read the dataframe: {e}")

        if index_col:
            df = Data.assign_index(data=df, index_column=index_col)

        if date_col:
            df = Data.assign_date(data=df, date_column=date_col, format=format)

        return df
    
    @staticmethod
    def assign_date(data:pd.DataFrame, date_column, format="%Y-%m-%d") -> pd.DataFrame:
        try:
            data[date_column] = pd.to_datetime(data[date_column], format=format)
        except Exception as e:
            msg = f"Could not assign the column: {date_column} as date column for: {e} error. Continuing without assigning."
            warnings.warn(msg, category=UserWarning)
            st.warning(msg)
        return data

    @staticmethod
    def handle_date_assignment(data: pd.DataFrame, date_column, numerical_columns, categorical_columns, format="%Y-%m-%d") -> tuple[pd.DataFrame, list, list, list]:
        """Attempts to set date; drops column from column lists if it succeeds."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            data = Data.assign_date(data=data, date_column=date_column, format=format)

        warning_occurred = any(
            issubclass(w.category, UserWarning) and "Could not assign the column" in str(w.message)
            for w in captured
        )

        if not warning_occurred:
            numerical_columns = [col for col in numerical_columns if col != date_column]
            categorical_columns = [col for col in categorical_columns if col != date_column]

        all_cols = numerical_columns + categorical_columns

        return data, numerical_columns, categorical_columns, all_cols

        
    @staticmethod
    def assign_index(data: pd.DataFrame, index_column) -> pd.DataFrame:
        try:
            data.set_index(index_column, inplace=True)
        except Exception as e:
            msg =  f"Could not assign the column: {index_column} as index column for: {e} error. Continuing without assigning."
            warnings.warn(msg, category=UserWarning)
            st.warning(msg)
        return data

    @staticmethod
    def handle_index_assignment(data: pd.DataFrame, index_column, numerical_columns, categorical_columns) -> tuple[pd.DataFrame, list, list, list]:
        """Attempts to set index; drops column from columns lists if it succeeds."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            data = Data.assign_index(data=data, index_column=index_column)

        warning_occurred = any(
            issubclass(w.category, UserWarning) and "Could not assign the column" in str(w.message)
            for w in captured
        )

        if warning_occurred:
            pass
        else:
            numerical_columns = [col for col in numerical_columns if col != index_column]
            categorical_columns = [col for col in categorical_columns if col != index_column]

        all_cols = numerical_columns + categorical_columns

        return data, numerical_columns, categorical_columns, all_cols

    @staticmethod
    def return_iqr(data, columns):
        if not isinstance(data, pd.DataFrame):
            st.error("DataFrame isn't a pandas DataFrame.")
            raise ValueError(f"DataFrame isn't a pandas DataFrame.")
            
        if isinstance(columns, str):
            columns = [columns]
        
        if not isinstance(columns, (list, tuple)):
            st.error(f"Given column: {columns} isn't in the waited types: [str, list, tuple]")
            return data
        
        data = data.copy()

        for col in columns:
            if col in data.columns:
                low, high = Data.calculate_iqr(data[col])
                data[col] = data[col].mask(
                    (data[col] < low) | (data[col] > high))
            else:
                st.warning(f"Column '{col}' not found in DataFrame.")
        return data

    @staticmethod
    def calculate_iqr(column):
        q_25 = column.quantile(0.25)
        q_75 = column.quantile(0.75)
        iqr = q_75 - q_25
        low = q_25 - (1.5 * iqr)
        high = q_75 + (1.5 * iqr)
        return (low, high)
