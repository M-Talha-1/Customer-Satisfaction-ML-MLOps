from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
# from steps.config import ModelNameConfig

@pipeline(enable_cache=True)
def train_pipeline():
    """
    Training pipeline to clean data, train a model and evaluate it.
    """
    df = ingest_data()
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train=X_train, y_train=y_train)
    mse, r2, rmse = evaluate_model(model=model, X_test=X_test, y_test=y_test)


    