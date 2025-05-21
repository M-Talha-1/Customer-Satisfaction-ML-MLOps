# Predicting Customer Satisfaction for Future Purchases (MLOps)


[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) [![MLflow](https://img.shields.io/badge/MLflow-v2.0.0-brightgreen.svg)](https://mlflow.org/) [![ZenML](https://img.shields.io/badge/ZenML-v0.20.0-orange.svg)](https://zenml.io/)

## 🚀 Project Overview

This project aims to predict the customer satisfaction score for the next order or purchase based on a customer's historical data. I use the **Brazilian E-Commerce Public Dataset by Olist** to analyze 100,000+ orders across various marketplaces from 2016 to 2018. By predicting satisfaction scores from features like order status, price, and reviews, the project demonstrates how to build production-ready ML pipelines using **ZenML** and **MLflow**.

The pipeline trains a machine learning model that predicts customer satisfaction and deploys it using MLflow, enabling continuous monitoring and deployment.

## 📄 Problem Statement

Given a customer's order history, we predict the review score for their next purchase based on features such as:

- **Order status**
- **Price**
- **Payment method**
- **Freight performance**
- **Customer reviews**

I create a **ZenML pipeline** that trains, evaluates, and deploys a machine learning model to predict customer satisfaction. By integrating **MLflow**, we can track model performance and ensure continuous deployment for future predictions.

## 🔧 Tools & Frameworks

- **ZenML**: Framework for building and deploying production-grade ML pipelines.
- **MLflow**: Tool for tracking experiments and model deployment.
- **Python 3.8+**: Core programming language for model development.


## 🧠 Project Workflow

1. **Data Ingestion**: Load and preprocess data from the Brazilian e-commerce dataset.
2. **Data Cleaning**: Remove unwanted columns and outliers.
3. **Model Training**: Train a machine learning model and log results using **MLflow** autologging.
4. **Model Evaluation**: Evaluate the model's performance based on the **Mean Squared Error (MSE)**.
5. **Model Deployment**: Deploy the model using **MLflow**, enabling real-time predictions.
6. **Continuous Deployment Pipeline**: Automate retraining and redeployment when model accuracy improves beyond a set threshold.

## 🛠 How to Run

Follow the steps below to clone and set up the repository:

```bash
# Clone the repository
git clone https://github.com/Mayankpratapsingh022/Customer-Satisfaction-MLOps.git

# Navigate to the project folder
cd Customer-Satisfaction-MLOps

# Install dependencies
pip install -r requirements.txt

# Set up ZenML with MLflow
pip install zenml[server]
zenml up
zenml integration install mlflow -y
```

## 📈 Training Pipeline

The training pipeline consists of several steps:

- **Ingest Data**: Load the dataset and create a pandas DataFrame.
- **Clean Data**: Perform data cleaning and feature selection.
- **Train Model**: Train a machine learning model and log performance with **MLflow autologging**.
- **Evaluate Model**: Calculate metrics such as **MSE** to evaluate the model.

## 🚀 Deployment Pipeline

The deployment pipeline handles continuous model deployment with **MLflow**:

1. **Deployment Trigger**: Check if the newly trained model meets accuracy requirements.
2. **Model Deployer**: Deploy the model if the evaluation metric passes the threshold.

The pipeline automatically updates the **MLflow** deployment server with the latest model version when performance improves.



## 📚 Resources & References

- **Dataset**: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/olistbr/brazilian-ecommerce)
- **ZenML Documentation**: [ZenML](https://docs.zenml.io/)
- **MLflow Documentation**: [MLflow](https://www.mlflow.org/)

## 🏆 Results

Using this framework, you can continuously predict customer satisfaction, deploy models in production, and monitor the performance for future improvements.

