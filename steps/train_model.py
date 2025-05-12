import logging

import pandas as pd
from zenml import step
from zenml.client import Client

from src.model_training import LinearRegressionModel
from sklearn.base import RegressorMixin
# from .config import ModelNameConfig

@step()
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RegressorMixin:
    """
    Trains the model
    Args:
        X_train: pandas DataFrame: Training data
        y_train: pandas Series: Training labels
    returns:
        LinearRegressionModel: Trained model
    """
    try:
        mlflow.sklearn.autolog()
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        logging.info("Model trained successfully")
        return trained_model

    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise e 