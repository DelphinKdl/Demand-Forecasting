from typing import Dict, Any
import pandas as pd
from common.utils import save_model


class PostprocessingPipeline:
    """
    Handles postprocessing steps in the machine learning pipeline.

    Responsibilities:
    - Saving the trained model after training
    - Formatting and returning prediction results during inference
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the postprocessing pipeline with configuration settings.

        Args:
            config (Dict[str, Any]): Dictionary containing all pipeline-related configuration.
        """
        self.config = config

    def run_train(self, model: Any) -> None:
        """
        Save the trained model to the file path specified in the config.

        Args:
            model (Any): Trained machine learning model.

        Returns:
            None
        """
        model_path = self.config['pipeline_runner']['model_path']
        save_model(model, base_path=model_path)

    def run_inference(self, y_pred: float, current_timestamp: pd.Timestamp) -> pd.DataFrame:
        """
        Format the model prediction as a single-row DataFrame for saving or further processing.

        This method:
        1. Accepts a prediction and a timestamp
        2. Constructs a DataFrame with these values

        Args:
            y_pred (float): Predicted value from the model.
            current_timestamp (pd.Timestamp): Timestamp of the prediction.

        Returns:
            pd.DataFrame: A single-row DataFrame with 'timestamp' and 'prediction' columns.
        """
        timestamp = pd.to_datetime(current_timestamp) + pd.Timedelta(self.config['pipeline_runner']['time_increment'])
        df_pred = pd.DataFrame({
            'datetime': [timestamp],
            'prediction': [y_pred]
        })
        return df_pred