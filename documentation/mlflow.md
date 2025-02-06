# MAUD MLFlow Documentation

## Introduction

MAUD uses MLFLow as the core provider of configuration management, tracking, tracing, and logging.

## Tracing

You'll see additiona decorators in some of the functions. This helps the Databricks agent framework trace out what is happening with the Gen AI stack.

## Configuration

We use the `mlflow.models.ModelConfig` class to manage configuration. This provides a 'development_config' parameter that can be used to specify a different configuration file for development purposes. Once we move this into model serving, this configuration is overwritten by the specified configuration file, which provides a very simple way to manage configuration for different environments.

## Logging

We use the `mlflow.log_metric` and `mlflow.log_param` functions to log metrics and parameters to MLFlow.

## Tracking

We use the `mlflow.log_metric` and `mlflow.log_param` functions to log metrics and parameters to MLFlow. We also use the 'mlflow.dspy.autolog' an mlflow.langchain.autolog'

