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
            categorical_columns = data.select_dtypes(include=["object", "category", "str"]).columns.to_list()

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
    def date_check(
        data: pd.DataFrame,
        date_column: str | int | list[str | int] | None = None,
    ) -> bool:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Provided data must be a pandas DataFrame.")

        if date_column is None:
            return False

        if isinstance(date_column, bool):
            raise TypeError("date_column cannot be a boolean.")

        if isinstance(date_column, (str, int)):
            requested_columns = [date_column]

        elif isinstance(date_column, list):
            if not date_column:
                raise ValueError("date_column cannot be an empty list.")

            requested_columns = date_column

        else:
            raise TypeError("date_column must be a string, integer position, "
                "list of strings/integers, or None."
                )
        
        requested_names = {item for item in requested_columns if isinstance(item, str)
        }

        column_positions: dict[str, list[int]] = {}

        if requested_names:
            for position, label in enumerate(data.columns):
                if label in requested_names:
                    column_positions.setdefault(label, []).append(position)

        type_errors = []
        value_errors = []

        resolved_columns = []

        for item in requested_columns:

            if isinstance(item, bool):
                type_errors.append(f"{item!r} is not a valid column reference; "
                    "boolean values are not supported."
                )
                continue

            if isinstance(item, int):
                if item < 0 or item >= data.shape[1]:
                    value_errors.append(f"Column position {item} is out of range. "
                        f"Valid positions are 0 to {data.shape[1] - 1}."
                    )
                    continue

                series = data.iloc[:, item]
                label = data.columns[item]

                resolved_columns.append((item, label, series))
                continue
        
            if isinstance(item, str):
                positions = column_positions.get(item, [])

                if not positions:
                    value_errors.append(f"Column {item!r} does not exist.")
                    continue

                if len(positions) > 1:
                    value_errors.append(
                        f"Column name {item!r} is ambiguous; "
                        f"it appears {len(positions)} times at "
                        f"positions {positions}.")
                    continue

                position = positions[0]
                series = data.iloc[:, position]
                label = data.columns[position]

                resolved_columns.append((item, label, series))
                continue

            type_errors.append(
                f"Invalid column reference {item!r}; "
                "expected a string or integer position."
            )

        for reference, label, series in resolved_columns:
            if not is_datetime64_any_dtype(series):
                value_errors.append(
                    f"Reference {reference!r} resolves to column "
                    f"{label!r}, but its dtype is not datetime-like."
                )

        if type_errors or value_errors:
            messages = []

            if type_errors:
                messages.append("Type errors:")
                messages.extend(f"- {message}"for message in type_errors)

            if value_errors:
                messages.append("Value errors:")
                messages.extend(f"- {message}" for message in value_errors)

            message = ("Invalid date_column specification:\n" + "\n".join(messages))
            
            if type_errors:
                raise TypeError(message)
            raise ValueError(message)

        return True