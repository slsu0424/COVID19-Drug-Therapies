# Databricks notebook source
# MAGIC %md #AutoML

# COMMAND ----------

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-databricks-automl-environment

# COMMAND ----------

pip install azureml-sdk[databricks]

# COMMAND ----------

df = spark.read.csv("/mnt/adls/FAERS_CSteroid_preprocess2.csv", header="true", nullValue = "NA", inferSchema="true")

display(df)

# COMMAND ----------

from sklearn.model_selection import train_test_split 
import pandas as pd

#df_train = spark.read.format("csv").load(f"abfss://{file_system_name}@{data_lake_account_name}.dfs.core.windows.net/DatasetDiabetes/preparedtraindata/",header=True,schema=transformedSchema)
#df_train = df_train.toPandas()

df_train = spark.read.csv("/mnt/adls/FAERS_CSteroid_preprocess2.csv", header="true", nullValue = "NA", inferSchema="true")
df_train = df_train.toPandas()

outcome_column = 'out_cod_DE'

#id_column = 'Id'
#df_train = df_train.drop(id_column,axis=1) 

# COMMAND ----------

import mlflow
import mlflow.azureml
import azureml.mlflow
import azureml.core

from azureml.core import Workspace

subscription_id = '9edd9e25-815c-4cdb-9bc8-d2ba127ec752'

# Azure Machine Learning resource group NOT the managed resource group
resource_group = '2021-06-22_SDUD' 

#Azure Machine Learning workspace name, NOT Azure Databricks workspace
workspace_name = 'SDUDaml'  

# Instantiate Azure Machine Learning workspace
ws = Workspace.get(name=workspace_name,
                   subscription_id=subscription_id,
                   resource_group=resource_group)

#Set MLflow experiment. 
#experimentName = "/Users/sansu@microsoft.com/Drug_Utilization/Notebook 3_FAERS" 
experimentName = "test2" 
mlflow.set_experiment(experimentName)

# COMMAND ----------

uri = ws.get_mlflow_tracking_uri()

print(uri)

mlflow.set_tracking_uri(uri)

# COMMAND ----------

# start a new MLflow training run

with mlflow.start_run():
#import mlflow

  # log metrics
  mlflow.log_metric("Accuracy", accuracy)
  mlflow.log_metric("Precision", precision)
  mlflow.log_metric("Recall", recall)
  mlflow.log_metric("F1-score", f1)
  #mlflow.log_metric("ROC AUC", auc)
  
  # log artifacts
  mlflow.log_artifact("confusion-matrix.png")
  #mlflow.log_artifact("roc-curve.png")
  
  # save the model to the outputs directory
  mlflow.sklearn.log_model(model, "model") # https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model
  
  # run id is logged as Azure ML Experiments Run ID
  print("Runid is:", mlflow.active_run().info.run_uuid)
  
  # end run
  mlflow.end_run()

# COMMAND ----------

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.core.dataset import Dataset

from azureml.train.automl.run import AutoMLRun
from azureml.train.automl import AutoMLConfig
from azureml.automl.runtime.onnx_convert import OnnxConverter
from azureml.core.model import Model
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice

ws = Workspace(workspace_name = workspace_name,
               subscription_id = subscription_id,
               resource_group = resource_group)
ws.write_config()   

# increase  the interation and experiment_timeout_hours as needed 
automl_settings = {
    "iterations": 20,
    "n_cross_validations": 5,
    "primary_metric": 'AUC_weighted',
    "enable_early_stopping": True,
    "max_concurrent_iterations": 5, 
    "model_explainability":True,
    "experiment_timeout_hours": 0.25
}
automl_config = AutoMLConfig(task = 'classification',
                             training_data = df_train,
                             label_column_name = 'is_readmitted',
                             **automl_settings
                            )
experiment = Experiment(ws, "COVIDPredictionExperiment")

# COMMAND ----------

local_run = experiment.submit(automl_config, show_output=True)
