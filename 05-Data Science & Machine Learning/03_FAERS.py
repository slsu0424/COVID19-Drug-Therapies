# Databricks notebook source
# MAGIC %md #Part 3: Baseline Modeling and Feature Engineering

# COMMAND ----------

# check python version

# https://vincentlauzon.com/2018/04/18/python-version-in-databricks
import sys

sys.version

# COMMAND ----------

pip install missingno pandasql pycaret[full]

# COMMAND ----------

# import libraries needed
import pandas as pd # for data analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns #for data visualization
from statistics import mode
import scipy as sp

# library settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from matplotlib.colors import ListedColormap

# COMMAND ----------

# MAGIC %md #Load data

# COMMAND ----------

df = spark.read.csv("/mnt/adls/FAERS_CSteroid_preprocess2.csv", header="true", nullValue = "NA", inferSchema="true")

display(df)

# COMMAND ----------

# how many rows, columns

print((df.count(), len(df.columns)))

# COMMAND ----------

# convert to pandas

df1 = df.toPandas()

# COMMAND ----------

df1.dtypes

# COMMAND ----------

# MAGIC %md ##Impute missing values

# COMMAND ----------

# MAGIC %md ###Numerical

# COMMAND ----------

# check null values in numerical variables 

df1.select_dtypes(exclude='object').isnull().sum()

# COMMAND ----------

# we are interested to impute for age, wt, dose_amt

# use mean (not too many outliers) or median (skewed distribution).  Median will be used due to skewed distributions.

# https://machinelearningbites.com/missing-values-imputation-strategies/
# https://vitalflux.com/imputing-missing-data-sklearn-simpleimputer/
# https://www.shanelynn.ie/pandas-iloc-loc-select-rows-and-columns-dataframe/#1-pandas-iloc-data-selection
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
# https://datascientyst.com/reshape-pandas-series-into-2d-array/

from sklearn.impute import SimpleImputer

df4 = df1.copy() 

imputer = SimpleImputer(missing_values=np.nan, strategy= 'median')

df4.age_in_yrs = imputer.fit_transform(df4['age_in_yrs'].values.reshape(-1,1)) # only convert age if age_cod is in years.
df4.wt_in_lbs = imputer.fit_transform(df4['wt_in_lbs'].values.reshape(-1,1))
df4.dose_amt = imputer.fit_transform(df4['dose_amt'].values.reshape(-1,1))

display(df4)

# COMMAND ----------

# MAGIC %md ###Categorical

# COMMAND ----------

# check null values in categorical variables 

df4.select_dtypes(include='object').isnull().sum()

# COMMAND ----------

# we are interested to impute for sex, route, dechal, dose_freq

# impute with most frequent
df5 = df4.copy()

imputer = SimpleImputer(missing_values=None, strategy= 'most_frequent')

df5.sex = imputer.fit_transform(df5['sex'].values.reshape(-1,1))
df5.route = imputer.fit_transform(df5['route'].values.reshape(-1,1))
df5.dechal = imputer.fit_transform(df5['dechal'].values.reshape(-1,1))
#df5.dose_freq = imputer.fit_transform(df5['dose_freq'].values.reshape(-1,1))

display(df5)

# COMMAND ----------

# inspect remaining missing values in data

import missingno as msno

msno.matrix(df5)

# COMMAND ----------

# MAGIC %md #Build a baseline model

# COMMAND ----------

#df5.select_dtypes(exclude='object').dtypes

# COMMAND ----------

# curate feature set (numerical values only)

#df6 = df5.copy()

#df6 = df5.select_dtypes(exclude='object')

df6 = df5.select_dtypes(exclude='object') \
                            .drop(['primaryid','caseid','caseversion','event_dt','mfr_dt','init_fda_dt','fda_dt','age','wt', \
                                    'rept_dt','last_case_version','val_vbm','start_dt','end_dt','drug_seq','dsg_drug_seq', 'nda_num'], axis=1)

# COMMAND ----------

display(df6)

# COMMAND ----------

# X = input
X = df6.drop('outc_cod_DE', axis= 1)

# y = output
y = df6['outc_cod_DE']

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# show size of each dataset (records, columns)
print("Dataset sizes: \nX_train", X_train.shape," \nX_test", X_test.shape, " \ny_train", y_train.shape, "\ny_test", y_test.shape)

data = {
    "train":{"X": X_train, "y": y_train},        
    "test":{"X": X_test, "y": y_test}
}

print ("Data contains", len(data['train']['X']), "training samples and",len(data['test']['X']), "test samples")

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

#from xgboost import XGBClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, classification_report

# COMMAND ----------

# https://ataspinar.com/2017/05/26/classification-with-scikit-learn/
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    #"QDA": QuadraticDiscriminantAnalysis(),
    "Gaussian Process": GaussianProcessClassifier() #http://www.ideal.ece.utexas.edu/seminar/GP-austin.pdf
}

def batch_classify(X_train, y_train, X_test, y_test, no_classifiers = 9, verbose = True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.
    
    Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train. 
    So it is best to train them on a smaller dataset first and 
    decide whether you want to comment them out or not based on the test accuracy score.
    """

# https://datascience.stackexchange.com/questions/28426/train-accuracy-vs-test-accuracy-vs-confusion-matrix

    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.process_time()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        t_end = time.process_time()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, y_train) # training accuracy
        test_score = classifier.score(X_test, y_test) # test accuracy
        precision = precision_score(y_test, predictions) # fraction of positive predictions that were correct. TP / (TP + FP)
        recall = recall_score(y_test, predictions) # fraction of positive predictions that were correctly identified.  TP / (TP + FN)
        f1 = f1_score(y_test, predictions) # avg of precision + recall ratios
        fbeta = fbeta_score(y_test, predictions, beta=0.5)
        class_report = classification_report(y_test, predictions)
        
        dict_models[classifier_name] = {'model': classifier, \
                                        'train_score': train_score, \
                                        'test_score': test_score, \
                                        'precision': precision_score, \
                                        'recall': recall_score, \
                                        'f1': f1, \
                                        'fbeta': fbeta, \
                                        'train_time': t_diff, 
                                        'class_report': class_report}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models


def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    precision_s = [dict_models[key]['precision'] for key in cls]
    recall_s = [dict_models[key]['recall'] for key in cls]
    f1_s = [dict_models[key]['f1'] for key in cls]
    fbeta_s = [dict_models[key]['fbeta'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    report = [dict_models[key]['class_report'] for key in cls]
    
    #df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),9)), columns = ['classifier', 'train_score', 'test_score', 'precision', 'recall', 'f1', 'fbeta', 'train_time', 'class_report'])
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),7)), columns = ['classifier', 'train_score', 'test_score', 'f1', 'fbeta', 'class_report', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'precision'] = precision_s[ii]
        df_.loc[ii, 'recall'] = recall_s[ii]
        df_.loc[ii, 'f1'] = f1_s[ii]
        df_.loc[ii, 'fbeta'] = fbeta_s[ii]
        df_.loc[ii, 'class_report'] = report[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    display(df_.sort_values(by=sort_by, ascending=False))

# COMMAND ----------

dict_models = batch_classify(X_train, y_train, X_test, y_test, no_classifiers = 10)

display_dict_models(dict_models)

# COMMAND ----------

# MAGIC %md #Feature Engineering

# COMMAND ----------

# MAGIC %md ##Encode categorical variables

# COMMAND ----------

# identify columns that are categorical and convert to numerical

# https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-numerical-variables/

df5.select_dtypes(include='object').head(5).T

# COMMAND ----------

# convert select categorical features to numerical as these will be useful features for modeling

# 2021-09-27 - Dropped reporter country as not relevant

df_converted = pd.get_dummies(df5, columns=['sex',
                                    'occr_country','role_cod','drugname','prod_ai','route','dechal','dose_freq','mfr_sndr'])

df_converted.head(2)

# COMMAND ----------

df_converted.select_dtypes(exclude='object').dtypes

# COMMAND ----------

df_converted.shape

# COMMAND ----------

# save data to ADLS Gen2

df_converted.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess3.csv', index=False)

# COMMAND ----------

# MAGIC %md ##Scale Data

# COMMAND ----------

# MAGIC %md ###Skew & Kurtosis

# COMMAND ----------

# MAGIC %md In order to prepare the data for machine learning tasks, we need to characterize the location and variability of the data.  A further characterization of the data includes data distribution, skewness and kurtosis.
# MAGIC 
# MAGIC Skewness - What is the shape of the distribution?  
# MAGIC 
# MAGIC Kurtosis - What is the measure of thickness or heaviness of the distribution?  
# MAGIC 
# MAGIC https://tekmarathon.com/2015/11/13/importance-of-data-distribution-in-training-machine-learning-models/

# COMMAND ----------

#pd.option_context('display.max_rows', None, 'display.max_columns', None)
#https://stackoverflow.com/questions/19124601/pretty-print-an-entire-pandas-series-dataframe
df5.dtypes

# COMMAND ----------

# drop data entry errors

# review selected numerical variables of interest
num_cols = ['age','wt','drug_seq','dose_amt','dsg_drug_seq']

plt.figure(figsize=(18,9))
df5[num_cols].boxplot()
plt.title("Numerical variables in the Corticosteroid dataset", fontsize=20)
plt.show()

# COMMAND ----------

df5['dose_amt'].max()

# COMMAND ----------

# inspect record

df5[df5['dose_amt'] == 30000].head(5)

# COMMAND ----------

# get all records with IU dose unit

df5[df5['dose_unit'] == 'IU'].head(10)

# COMMAND ----------

# this looks to be an outlier, drop this record

# https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/

df5.drop(df5[df5['dose_amt'] == 30000].index, inplace=True)

# COMMAND ----------

# visualize skew for a sample feature

# https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45
# https://www.analyticsvidhya.com/blog/2021/05/shape-of-data-skewness-and-kurtosis/
# https://opendatascience.com/transforming-skewed-data-for-machine-learning/

sns.distplot(df5['dose_amt'])

# COMMAND ----------

# calculate skew value

# https://www.geeksforgeeks.org/scipy-stats-skew-python/
# skewness > 0 : more weight in the left tail of the distribution.  

# https://medium.com/@TheDataGyan/day-8-data-transformation-skewness-normalization-and-much-more-4c144d370e55
# If skewness value lies above +1 or below -1, data is highly skewed. If it lies between +0.5 to -0.5, it is moderately skewed. If the value is 0, then the data is symmetric

# https://vivekrai1011.medium.com/skewness-and-kurtosis-in-machine-learning-c19f79e2d7a5
# If the peak of the distribution is in right side that means our data is negatively skewed and most of the people reported with AEs weigh more than the average.

df5['dose_amt'].skew()

# COMMAND ----------

# calculate kurtosis value

df5['dose_amt'].kurtosis() # platykurtic distribution (low degree of peakedness)

# COMMAND ----------

# MAGIC %md ###Log Transformation

# COMMAND ----------

# convert dataframe columns to list

# https://datatofish.com/convert-pandas-dataframe-to-list

num_col = df5.select_dtypes(exclude='object') \
              .columns.drop(['outc_cod_DE']) \
              .values.tolist()

print(num_col)

# COMMAND ----------

# Remove skewnewss and kurtosis using log transformation if it is above a threshold value (2)

# https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45

# create empty pandas dataframe
statdataframe = pd.DataFrame()

# assign num_col values to new 'numeric column' in statdataframe
statdataframe['numeric_column'] = num_col

skew_before = []
skew_after = []

kurt_before = []
kurt_after = []

standard_deviation_before = []
standard_deviation_after = []

log_transform_needed = []
log_type = []

for i in num_col:
    skewval = df5[i].skew()
    skew_before.append(skewval)
    
    kurtval = df5[i].kurtosis()
    kurt_before.append(kurtval)
    
    sdval = df5[i].std()
    standard_deviation_before.append(sdval)
    
    # https://quick-adviser.com/what-does-the-kurtosis-value-tell-us
    if (abs(skewval) >2) & (abs(kurtval) >2):
        log_transform_needed.append('Yes')
        
        # are there any features that have values of 0 (no predictive power)?
        if len(df5[df5[i] == 0])/len(df5) <= 0.02:
            log_type.append('log')
            skewvalnew = np.log(pd.DataFrame(df5[df5[i] > 0])[i]).skew()
            skew_after.append(skewvalnew)
            
            kurtvalnew = np.log(pd.DataFrame(df5[df5[i] > 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)
            
            sdvalnew = np.log(pd.DataFrame(df5[df5[i] > 0])[i]).std()
            standard_deviation_after.append(sdvalnew)
            
    else:
        log_type.append('NA')
        log_transform_needed.append('No')
        
        skew_after.append(skewval)
        kurt_after.append(kurtval)
        standard_deviation_after.append(sdval)

statdataframe['skew_before'] = skew_before
statdataframe['kurtosis_before'] = kurt_before
statdataframe['standard_deviation_before'] = standard_deviation_before
statdataframe['log_transform_needed'] = log_transform_needed
statdataframe['log_type'] = log_type
statdataframe['skew_after'] = skew_after
statdataframe['kurtosis_after'] = kurt_after
statdataframe['standard_deviation_after'] = standard_deviation_after

statdataframe

# COMMAND ----------

# perform log transformation for the columns that need it above

# only perform log transformation for dose_amt

for i in range(len(statdataframe)):
    if statdataframe['log_transform_needed'][i] == 'Yes':
        colname = str(statdataframe['numeric_column'][i])
        
        if statdataframe['numeric_column'][i] == 'dose_amt':
            df5 = df4[df4[colname] > 0]
            df5[colname + "_log"] = np.log(df5[colname]) 

# COMMAND ----------

# drop original columns that needed log transformation and use log counterparts

df5 = df5.drop(['dose_amt'], axis = 1)

# COMMAND ----------

# update list with numeric features (after log transformation)

numerics = list(set(list(df5._get_numeric_data().columns))- {'outc_cod_DE'})

numerics

# COMMAND ----------

# MAGIC %md ##Scale data  
# MAGIC 
# MAGIC Standardization comes into picture when features of input data set have large differences between their ranges, or simply when they are measured in different measurement units (e.g., Pounds, Meters, Miles … etc)
# MAGIC 
# MAGIC https://builtin.com/data-science/when-and-why-standardize-your-data
# MAGIC https://www.researchgate.net/post/log_transformation_and_standardization_which_should_come_first

# COMMAND ----------

# check stats on numeric features

datf = pd.DataFrame()

datf['features'] = numerics
datf['std_dev'] = datf['features'].apply(lambda x: df5[x].std())
datf['mean'] = datf['features'].apply(lambda x: df5[x].mean())

print(datf)

# COMMAND ----------

# standardize function

def standardize(raw_data):
    return ((raw_data - np.mean(raw_data, axis = 0)) / np.std(raw_data, axis = 0))
  
# standardize the data 
df5[numerics] = standardize(df5[numerics])

# COMMAND ----------

df5.display(5)

# COMMAND ----------

# MAGIC %md ##Remove outliers

# COMMAND ----------

df3.dtypes

# COMMAND ----------

# detect outliers

# https://www.kaggle.com/agrawaladitya/step-by-step-data-preprocessing-eda
# https://www.machinelearningplus.com/plots/python-boxplot

# select numerical variables of interest
num_cols = ['age_in_yrs','wt_in_lbs','drug_seq','dose_amt','dsg_drug_seq']
#num_cols = ['age_in_yrs','drug_seq','dose_amt','dsg_drug_seq']

plt.figure(figsize=(18,9))
df3[num_cols].boxplot()
plt.title("Numerical variables in the Corticosteroids dataset", fontsize=20)
plt.show()

# COMMAND ----------

# MAGIC %md ##Interaction variables

# COMMAND ----------

# https://pycaret.org
# https://pycaret.readthedocs.io/en/latest/api/classification.html
# https://github.com/pycaret/pycaret/blob/master/examples/PyCaret%202%20Classification.ipynb

# Importing module and initializing setup

#from pycaret.classification import *
#reg1 = setup(data = df6, target = 'outc_cod_DE')

# COMMAND ----------

# spot inspect the data

# results show the need to consolidate the PT terms

df3[df3['caseid'] == 17639954].head(5)

# COMMAND ----------

# find all rows with null values

# https://datatofish.com/rows-with-nan-pandas-dataframe/

print(df3[df3.isnull().any(axis=1)])
