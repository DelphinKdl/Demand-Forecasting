import os
import sys
from pathlib import Path

import pandas as pd
from typing import Dict, Any



class DataManager:
    """
    A utility class responsible for handling all data-related I/O operations
    and transformations used across the ML pipeline.

    Responsibilities:
    - Initializing production database
    - Loading and saving parquet files
    - Appending new data to existing datasets
    - Slicing or filtering data by timestamp
    - Saving predictions incrementally
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataManager with a configuration dictionary.

        Args:
            config (Dict[str, Any]): Configuration parameters for paths and filenames.
        """
        self.config = config

    def initialize_prod_database(self) -> None:
        """
        Initialize the production database by copying the raw database
        into the production folder.

        Returns:
            None
        """
        raw_data_path = os.path.join(
            self.config['data_manager']['raw_data_folder'],
            self.config['data_manager']['raw_database_name']
        )
        prod_data_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['prod_database_name']
        )
        df = pd.read_parquet(raw_data_path)
        # Save the data to the prod folder to initialize production "database"
        df.to_parquet(prod_data_path, index=False)

        # If the prediction file exist from the previous runs, we delete it
        prediction_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['real_time_prediction_data_name']
        )
        if os.path.exists(prediction_path):
            os.remove(prediction_path)

    @staticmethod
    def append_data(current_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Append new data to an existing DataFrame and reset the index.

        Args:
            current_data (pd.DataFrame): The existing base data.
            new_data (pd.DataFrame): The new data to append.

        Returns:
            pd.DataFrame: Combined DataFrame with continuous index.
        """
        df = pd.concat([current_data, new_data], axis=0)
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def get_n_last_points(data: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Retrieve the last `n` rows from a DataFrame.

        Args:
            data (pd.DataFrame): Source DataFrame.
            n (int): Number of rows to retrieve from the end.

        Returns:
            pd.DataFrame: The last `n` rows.
        """
        return data.iloc[-n:].copy()

    @staticmethod
    def get_timestamp_data(data: pd.DataFrame, timestamp: str) -> pd.DataFrame:
        """
        Filter the DataFrame to get all rows with a matching datetime.

        Args:
            data (pd.DataFrame): DataFrame containing a 'datetime' column.
            timestamp (str): Timestamp string to match.

        Returns:
            pd.DataFrame: Filtered rows where 'datetime' matches the input timestamp.
        """
        return data.loc[pd.to_datetime(data['datetime']) == pd.to_datetime(timestamp)].copy()

    @staticmethod
    def load_data(path: str) -> pd.DataFrame:
        """
        Load a DataFrame from a parquet file.

        Args:
            path (str): Path to the parquet file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        return pd.read_parquet(path)

    @staticmethod
    def save_data(data: pd.DataFrame, path: str) -> None:
        """
        Save a DataFrame to a CSV file.

        Args:
            data (pd.DataFrame): Data to be saved.
            path (str): Output file path.

        Returns:
            None
        """
        data.to_parquet(path, index=False)

    def save_predictions(self, df_pred: pd.DataFrame, current_timestamp: pd.Timestamp) -> None:
        """
        Save predictions to the production CSV file.
        Appends to the existing file unless it's the first timestamp, in which case it overwrites.

        Args:
            df_pred (pd.DataFrame): Single-row DataFrame with prediction and timestamp.
            current_timestamp (pd.Timestamp): Timestamp used to determine whether to append or overwrite.

        Returns:
            None
        """
        prediction_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['real_time_prediction_data_name']
        )

        # Determine whether to overwrite or append
        if os.path.exists(prediction_path):
            if current_timestamp == pd.to_datetime(self.config['pipeline_runner']['first_timestamp']):
                # Start fresh for first timestamp
                combined_df = df_pred
            else:
                # Append to existing predictions
                existing_pred_df = pd.read_parquet(prediction_path)
                combined_df = pd.concat([existing_pred_df, df_pred], ignore_index=True)
        else:
            # File doesn't exist yet
            combined_df = df_pred

        # Save final DataFrame
        combined_df.to_parquet(prediction_path, index=False)

    def load_prod_data(self) -> pd.DataFrame:
        """
        Load the production data (true values) from the configured file, always parsing 'datetime'.
        Returns:
            pd.DataFrame: Loaded production data with 'datetime' parsed.
        """
        prod_data_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['prod_database_name']
        )
        df = self.load_data(prod_data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df

    def load_prediction_data(self) -> pd.DataFrame:
        """
        Load the real-time prediction data from the configured file, always parsing 'datetime'.
        Returns:
            pd.DataFrame: Loaded prediction data with 'datetime' parsed.
        """
        prediction_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['real_time_prediction_data_name']
        )
        df = self.load_data(prediction_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df 