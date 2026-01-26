import os
import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

project_root = Path(__file__).resolve().parent.parent.parent.parent # Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' /'src'))

import pandas as pd
from typing import Dict, Any
from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline
from pipelines.training import TrainingPipeline
from pipelines.inference import InferencePipeline
from pipelines.postprocessing import PostprocessingPipeline


class PipelineRunner:
    """
    A class that orchestrates the execution of all stages in the ML pipeline.

    This includes:
    - Preprocessing
    - Feature engineering
    - Training
    - Inference
    - Postprocessing

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        data_manager (DataManager): Manages loading/saving and transformation of data.
        real_time_data (pd.DataFrame): Cached real-time production data for inference.
        current_database_data (pd.DataFrame): Cached production database data for inference.
        prod_data_path (str): Path to the production database file.
        preprocessing_pipeline (PreprocessingPipeline): Handles data preprocessing steps.
        feature_eng_pipeline (FeatureEngineeringPipeline): Handles feature engineering steps.
        training_pipeline (TrainingPipeline): Handles model training steps.
        inference_pipeline (InferencePipeline): Handles inference steps.
        postprocessing_pipeline (PostprocessingPipeline): Handles postprocessing steps.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initialize the pipeline runner and its pipeline components.

        Args:
            config (Dict[str, Any]): Dictionary containing all pipeline configurations.
            data_manager (DataManager): Instance for managing I/O operations on data.
        """
        self.config = config
        self.data_manager = data_manager

        # Initialize individual pipeline components
        self.preprocessing_pipeline = PreprocessingPipeline(config=config)
        self.feature_eng_pipeline = FeatureEngineeringPipeline(config=config)
        self.training_pipeline = TrainingPipeline(config=config)
        self.inference_pipeline = InferencePipeline(config=config)
        self.postprocessing_pipeline = PostprocessingPipeline(config=config)

        # Load real-time data
        self.real_time_data = self.data_manager.load_data(
            os.path.join(
                config['data_manager']['prod_data_folder'],
                config['data_manager']['real_time_data_prod_name']
            )
        )

        # Path to current production database
        self.prod_data_path = os.path.join(
            self.config['data_manager']['prod_data_folder'],
            self.config['data_manager']['prod_database_name']
        )

        # Load existing production database
        self.current_database_data = self.data_manager.load_data(self.prod_data_path)

    def run_training(self) -> None:
        """
        Run the full training pipeline:
        1. Load and preprocess data
        2. Perform feature engineering
        3. Train the model
        4. Save the trained model

        Returns:
            None
        """
        df = self.data_manager.load_data(self.prod_data_path)
        df = self.preprocessing_pipeline.run(df=df)
        df = self.feature_eng_pipeline.run(df=df)
        model = self.training_pipeline.run(df)
        self.postprocessing_pipeline.run_train(model=model)
        return

    def run_inference(self, current_timestamp: pd.Timestamp) -> None:
        """
        Run the full inference pipeline:
        1. Load real-time data for the current timestamp
        2. Append to the production database
        3. Prepare the latest batch
        4. Preprocess, transform, and predict
        5. Postprocess and store the prediction
        6. Update the production database

        Args:
            current_timestamp (pd.Timestamp): The timestamp for which to run inference.

        Returns:
            None
        """

        # Step 1: Retrieve real-time data for the current timestamp
        current_real_time_data = self.data_manager.get_timestamp_data(
            data=self.real_time_data,
            timestamp=current_timestamp
        )

        # Step 2: Append new data to production database
        self.current_database_data = self.data_manager.append_data(
            current_data=self.current_database_data,
            new_data=current_real_time_data
        )

        # Step 3: Get the last N rows as the latest batch
        df = self.data_manager.get_n_last_points(
            data=self.current_database_data,
            n=self.config['pipeline_runner']['batch_size']
        )

        # Step 4: Run preprocessing and feature engineering
        df = self.preprocessing_pipeline.run(df=df)
        df = self.feature_eng_pipeline.run(df=df)

        # Step 5: Run inference
        y_pred = self.inference_pipeline.run(x=df)

        # Step 6: Postprocessing and saving the prediction
        df_pred = self.postprocessing_pipeline.run_inference(
            y_pred=y_pred,
            current_timestamp=current_timestamp
        )
        # Step 7: Save the prediction and updated database to access in the UI application
        self.data_manager.save_predictions(df_pred, current_timestamp)
        self.data_manager.save_data(data=self.current_database_data, path=self.prod_data_path)
        return