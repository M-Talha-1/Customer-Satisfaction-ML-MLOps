import logging
from zenml.steps import step

import pandas as pd
from src.model_evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

@step()
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
    ) -> Tuple[
        Annotated[float, "Mean Squared Error"],
        Annotated[float, "R2 Score"],
        Annotated[float, "Root Mean Squared Error"]
    ]:
    """
    Evaluate the model using the provided DataFrame.

    Args:
        model: regressionMixin: Trained model
    """
    try:
        predictions = model.predict(X_test)
        mse_eval = MSE()
        mse = mse_eval.calculate_score(y_test, predictions) 

        r2_eval = R2()
        r2 = r2_eval.calculate_score(y_test, predictions)

        rmse_eval = RMSE()
        rmse = rmse_eval.calculate_score(y_test, predictions)
        
        return mse, r2, rmse
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e