# Databricks notebook source
# MAGIC %md #Part 2: Exploratory Data Analysis & Preprocessing for AutoML

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
