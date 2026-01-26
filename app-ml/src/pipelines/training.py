import pandas as pd
import numpy as np
import optuna
from typing import Dict, Tuple, Any
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

class TrainingPipeline:
    """
    A pipeline class for training and optimizing machine learning models,
    specifically CatBoost, using configuration-driven parameters and Optuna for tuning.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary with training parameters.
        optuna_config (Dict[str, Any]): Subset of config containing Optuna-specific settings.
        search_space (Dict[str, Any]): Hyperparameter search space for tuning.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the TrainingPipeline with the provided configuration.

        Args:
            config (Dict[str, Any]): Full pipeline configuration dictionary.
        """
        self.config: Dict[str, Any] = config['training']
        self.optuna_config: Dict[str, Any] = self.config.get('optuna', {})
        self.search_space: Dict[str, Any] = self.config['optuna']['search_space']

    @staticmethod
    def make_target(df: pd.DataFrame, target_params: Dict[str, str]) -> pd.DataFrame:
        """
        Create a shifted target column for forecasting tasks.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_params (Dict[str, str]): Parameters dict with keys:
                - 'target_column': source column name
                - 'shift_period': how far to shift the target forward
                - 'new_target_name': name of the resulting target column

        Returns:
            pd.DataFrame: DataFrame with a new target column.
        """
        shift_period = target_params['shift_period']
        df[target_params['new_target_name']] = df[target_params['target_column']].shift(-shift_period).ffill()
        return df

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepares training and testing datasets by applying a target transformation
        and splitting by fraction (no shuffling).

        Args:
            df (pd.DataFrame): Input DataFrame with features and target.

        Returns:
            Tuple containing:
                - x_train (pd.DataFrame): Training features
                - x_test (pd.DataFrame): Testing features
                - y_train (pd.Series): Training target
                - y_test (pd.Series): Testing target
        """
        df = self.make_target(df, target_params=self.config['target_params'])
        feats = [col for col in df.columns if col != self.config['target_params']['new_target_name']]
        x, y = df[feats], df[self.config['target_params']['new_target_name']]

        train_size = int(self.config['train_fraction'] * len(df))
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return x_train, x_test, y_train, y_test

    def tune_hyperparams(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple[Any, optuna.Study]:
        """
        Perform hyperparameter tuning using Optuna, then retrain the model
        using the best configuration on the full training data.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            x_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test targets.

        Returns:
            Tuple containing:
                - Trained CatBoost model with best parameters
                - Completed Optuna Study object
        """
        np.random.seed(42)

        def objective(trial: optuna.Trial) -> float:
            ss = self.search_space
            params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", ss["learning_rate"]["low"], ss["learning_rate"]["high"],
                    log=ss["learning_rate"].get("log", False)
                ),
                "depth": trial.suggest_int("depth", ss["depth"]["low"], ss["depth"]["high"]),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", ss["l2_leaf_reg"]["low"], ss["l2_leaf_reg"]["high"],
                    log=ss["l2_leaf_reg"].get("log", False)
                ),
                "iterations": self.config["iterations"],
                "loss_function": self.config["loss_function"],
                "verbose": self.config.get("verbose", 0)
            }

            # Manual time-based validation split
            train_idx = int(self.config['train_fraction'] * len(x_train))
            x_tr, x_val = x_train.iloc[:train_idx], x_train.iloc[train_idx:]
            y_tr, y_val = y_train.iloc[:train_idx], y_train.iloc[train_idx:]

            model = CatBoostRegressor(**params, random_seed=42, allow_writing_files=False)
            model.fit(
                x_tr, y_tr,
                eval_set=(x_val, y_val),
                early_stopping_rounds=self.config.get("early_stopping_rounds", 100),
                use_best_model=True,
                verbose=False

            )

            preds = model.predict(x_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            trial.set_user_attr("best_iteration", model.get_best_iteration())
            return rmse

        # Run Optuna study
        n_trials = self.optuna_config["n_trials"]
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        # Train final model on full training data with best parameters
        best_params = study.best_params.copy()
        best_params.update({
            "iterations": study.best_trial.user_attrs["best_iteration"],
            "loss_function": self.config["loss_function"],
            "verbose": False
        })

        # Concatenate training and testing data
        x_train_test = pd.concat([x_train, x_test], axis=0)
        y_train_test = pd.concat([y_train, y_test], axis=0)

        final_model = CatBoostRegressor(**best_params, random_seed=42, allow_writing_files=False)
        final_model.fit(x_train_test, y_train_test, verbose=False)

        return final_model, study

    def run(self, df: pd.DataFrame) -> Any:
        """
        Run the full training pipeline:
        1. Generate target column
        2. Train-test split
        3. Tune hyperparameters with Optuna
        4. Train final model using best parameters

        Args:
            df (pd.DataFrame): Input training DataFrame with features and target.

        Returns:
            Any: Trained model (e.g., CatBoostRegressor)
        """
        x_train, x_test, y_train, y_test = self.prepare_dataset(df)
        model, _ = self.tune_hyperparams(x_train, y_train, x_test, y_test)
        return model