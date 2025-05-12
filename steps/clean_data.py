import logging

import pandas as pd
from zenml import step
from src.data_cleaning import DataProcessStrategy, DataSplitStrategy, DataCleaning
from typing_extensions import Annotated
from typing import Tuple
import numpy as np

@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Clean data and split it into train and test data
    Args:
        data: pd.DataFrame: raw data
    return:
        X_train: pd.DataFrame: training data
        y_train: pd.Series: training target
        X_test: pd.DataFrame: testing data
        y_test: pd.Series: testing target
    """
    try:
        data_process = DataProcessStrategy()
        data_cleaning = DataCleaning(data, data_process)
        processed_data = data_cleaning.handle_data()

        data_split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(processed_data, data_split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in data cleaning {}".format(e))
        raise e
    

