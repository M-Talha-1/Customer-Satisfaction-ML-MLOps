import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataProcessStrategy(DataStrategy):
    """
    Concrete class implementing strategy for handling data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method to handle data
        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
                logging.error(e)
                raise e
        
class DataSplitStrategy(DataStrategy):
     """
     Strategy to split data into train and test
     """

     def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
            """
            Method to split data into train and test
            :param data: pd.DataFrame
            :return: pd.DataFrame
            """
            
            try:
                X = data.drop("review_score", axis=1)
                y = data["review_score"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                return X_train, X_test, y_train, y_test
            except Exception as e:
                logging.error("Error in spliting data {}".format(e))
                raise e

class DataCleaning:
    """
    Class to clean data which proceesses the data and divides it into train and test data
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)
