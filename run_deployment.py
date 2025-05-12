from typing import cast

import click
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    # inference_pipeline,
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose to only run deployment (`deploy`), only prediction (`predict`), or both (`deploy_and_predict`).",
)
@click.option(
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy required to deploy the model",
)
def run_deployment(config: str, min_accuracy: float):
    """Run the MLflow deployment and/or inference pipeline."""
    
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        # Run deployment pipeline
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60,
        )

    if predict:
        # Run inference pipeline
        # inference_pipeline()
        pass

    print(
        "You can run:\n "
        f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'"
        "[/italic green]\n...to inspect your experiment runs within the MLflow UI."
    )

    # Get deployed MLflow model server
    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    services = mlflow_model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if services:
        service = cast(MLFlowDeploymentService, services[0])

        if service.is_running:
            print(
                f"The MLflow prediction server is running at:\n"
                f"    {service.prediction_url}\n"
                f"To stop it, run:\n"
                f"[italic green]zenml model-deployer models delete {service.uuid}[/italic green]"
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f"State: {service.status.state.value}\n"
                f"Error: {service.status.last_error}"
            )
        else:
            print("MLflow service is not currently running.")
    else:
        print(
            "No MLflow prediction server found. Please deploy a model first using `--deploy`."
        )

if __name__ == "__main__":
    run_deployment()
