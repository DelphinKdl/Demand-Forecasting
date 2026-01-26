import pandas as pd
from typing import Dict, Any
from common.utils import load_model

class InferencePipeline:
    """
    A pipeline for making predictions using a trained model.

    This class handles:
    - Loading a trained model
    - Preparing input data for inference
    - Making predictions
    - Post-processing predictions

    Args:
        config (Dict[str, Any]): Configuration dictionary containing inference parameters
    """
    def __init__(self, config: Dict[str, Any]) -> object:
        """
        Initializes the InferencePipeline with configuration data.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing inference parameters
        """
        self.config = config

    def run(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete inference pipeline.

        This method:
        1. Makes predictions using the loaded model
        2. Gets current timestamp
        3. Passes predictions and timestamp to postprocessing pipeline

        Args:
            x (pd.DataFrame): Input DataFrame containing features for prediction

        Returns:
            pd.DataFrame: The last prediction value from the model
        """
        # Load the model
        model = load_model(base_path=self.config['pipeline_runner']['model_path'])
        # Make prediction
        y_pred = model.predict(x)
        # Take the last point prediction only
        y_pred = y_pred[-1]
        return y_pred