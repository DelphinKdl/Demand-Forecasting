import pandas as pd
from typing import Dict, List


class PreprocessingPipeline:
    """
    A pipeline for preprocessing the raw data.

    This class handles the preprocessing steps including:
    - Column renaming
    - Column dropping

    Args:
        config (Dict[str, str]): Configuration dictionary containing preprocessing parameters
    """
    def __init__(self, config: Dict[str, str]):
        self.config = config['preprocessing']

    @staticmethod
    def rename_columns(df, column_mapping):
        """
        Rename columns in the dataset using a mapping dictionary.

        Args:
            df (pd.DataFrame): Input DataFrame
            column_mapping (dict): Dictionary mapping old column names to new column names.
                                 Example: {'old_name': 'new_name'}

        Returns:
            pd.DataFrame: DataFrame with renamed columns
        """
        return df.rename(columns=column_mapping)

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): List of column names to drop

        Returns:
            pd.DataFrame: DataFrame with specified columns removed
        """
        df.drop(columns=columns, inplace=True)
        return df

    def run(self, df: pd.DataFrame):
        """
        Execute the complete preprocessing pipeline on the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to be preprocessed

        Returns:
            pd.DataFrame: Preprocessed DataFrame with renamed columns, dropped columns, and target variable
        """
        df.reset_index(drop=True, inplace=True)
        df = self.rename_columns(df, self.config['column_mapping'])
        df = self.drop_columns(df, self.config['drop_columns'])
        return df