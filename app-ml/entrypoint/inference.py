"""
Inference Pipeline:
- Loads configuration
- Initializes production database
- Loops through a given number of timestamps and runs inference
- Saves predictions
- Plots comparison of predicted vs actual values
"""

import os
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
os.chdir(project_root)
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' /'src'))

from common.utils import read_config, plot_predictions_vs_actual
from pipelines.pipeline_runner import PipelineRunner
from common.data_manager import DataManager


if __name__ == "__main__":
    # Number of runs for
    num_timestamps = 200

    # Load config file
    config_path = project_root / 'config' / 'config.yaml'
    config = read_config(config_path)

    # Initialize data manager and timestamps
    data_manager = DataManager(config)
    current_timestamp = pd.to_datetime(config['pipeline_runner']['first_timestamp'])
    time_increment = pd.Timedelta(config['pipeline_runner']['time_increment'])

    # Prepare production database
    data_manager.initialize_prod_database()

    # Initialize Pipeline Runner
    pipeline_runner = PipelineRunner(config=config, data_manager=data_manager)
    
    # Load the dataset to run inference on
    dataset_path = os.path.join(
        config['data_manager']['prod_data_folder'],
        config['data_manager']['real_time_data_prod_name']
    )
    df = data_manager.load_data(dataset_path)

    # Loop through timestamps
    for i in range(num_timestamps):
        print(f"Processing timestamp {i+1}/{num_timestamps}: {current_timestamp}")

        # Run inference on this timestamp's data
        pipeline_runner.run_inference(current_timestamp)
        
        # Increment timestamp
        current_timestamp += time_increment
    
    print("Inference completed for all timestamps!")
    
    # Load predictions and actual data for plotting
    print("Loading data for plotting...")
    predictions_df = data_manager.load_prediction_data()
    actual_df = data_manager.load_prod_data()
    plot_predictions_vs_actual(predictions_df, actual_df)