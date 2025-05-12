import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: pandas DataFrame: Training data
            y_train: pandas Series: Training labels
        returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model
    """
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: pandas DataFrame: Training data
            y_train: pandas Series: Training labels
        returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model trained successfully")
            return reg
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise e