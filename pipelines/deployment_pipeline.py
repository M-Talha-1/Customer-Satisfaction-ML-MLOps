import pandas as pd
import numpy as np

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pydantic import BaseModel

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model

# Use Docker image with MLflow integration
docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):
    """Parameters that are used to trigger the deployment."""
    min_accuracy: float = 0.9

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Triggers model deployment if accuracy > threshold."""
    return accuracy > config.min_accuracy

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Data ingestion and preparation
    df = ingest_data()
    X_train, X_test, y_train, y_test = clean_data(df)

    # Train and evaluate model
    model = train_model(X_train=X_train, y_train=y_train)
    mse, r2, rmse = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    # Trigger deployment if accuracy is good
    deployment_decision = deployment_trigger(accuracy=mse, config=DeploymentTriggerConfig(min_accuracy=min_accuracy))

    # MLflow model deployment
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )
