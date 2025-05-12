import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):
    """
    Abstract class defining strategy for model evaluation
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the score of the model
        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        returns:
            float: Score of the model
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error
        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        returns:
            None
        """
        try:
            logging.info("Calculating Mean Squared Error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the R2 Score
        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        returns:
            None
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2 Score: {e}")
            raise e

class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Root Mean Squared Error
        Args:
            y_true: np.ndarray: True labels
            y_pred: np.ndarray: Predicted labels
        returns:
            None
        """
        try:
            logging.info("Calculating Root Mean Squared Error")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e

