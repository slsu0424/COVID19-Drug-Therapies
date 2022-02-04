# Databricks notebook source
# MAGIC %md #Part 2: Preprocessing & Feature Engineering

# COMMAND ----------

# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml

# https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114#:~:text=Logarithm%20transformation%20%28or%20log%20transform%29%20is%20one%20of,transformation%2C%20the%20distribution%20becomes%20more%20approximate%20to%20normal.

# COMMAND ----------

# check python version

# https://vincentlauzon.com/2018/04/18/python-version-in-databricks
import sys

sys.version

# COMMAND ----------

pip install missingno pandasql

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

df = spark.read.csv("/mnt/adls/FAERS_CSteroid_preprocess1.csv", header="true", nullValue = "NA", inferSchema="true")

display(df)

# COMMAND ----------

# how many rows, columns

print((df.count(), len(df.columns)))

# COMMAND ----------

# MAGIC %md #Preprocess data

# COMMAND ----------

# MAGIC %md ##Create target variable

# COMMAND ----------

# convert to pandas

df1 = df.toPandas()

# COMMAND ----------

# new feature - outcome code
# add as a new column

df1.insert(loc = 52, 
          column = 'outc_cod_DE', 
          value = 0)

display(df1)

# COMMAND ----------

# target will be binary classification - did a patient die?

df1['outc_cod_DE'] = np.where(df1['outc_cod']=='DE',1,0)

# COMMAND ----------

df1['outc_cod_DE'].value_counts()

# COMMAND ----------

# MAGIC %md ###Remove dup caseid

# COMMAND ----------

# merge dup caseid where outc_cod = DE is present

# return all record where outcome of death = 1

df1.loc[df1['outc_cod_DE'] == 1].head(5)

# COMMAND ----------

# collapse multiple caseid (FAERS case)

# where there are multiple AEs for the same caseid, collapse these records by taking the max record where outc_code = DE (1)

#https://stackoverflow.com/questions/49343860/how-to-drop-duplicate-values-based-in-specific-columns-using-pandas

df2 = df1.sort_values(by=['outc_cod_DE'], ascending = False) \
        .drop_duplicates(subset=['caseid'], keep = 'first')

# COMMAND ----------

# re-inspect same case

df2[df2['caseid'] == 15954362].head(5)

# COMMAND ----------

# how many records after removing dups
df2.shape

# COMMAND ----------

# count distinct cases

# https://datascienceparichay.com/article/pandas-count-of-unique-values-in-each-column

print(df2.nunique())

#from pyspark.sql.functions import countDistinct

#df4.select(countDistinct("caseid")).show()

# COMMAND ----------

# MAGIC %md ###Check for imbalance

# COMMAND ----------

# check label ratios
# https://dataaspirant.com/handle-imbalanced-data-machine-learning/

sns.countplot(x='outc_cod_DE', data=df2) # data already looks wildly imbalanced but let us continue

# COMMAND ----------

# MAGIC %md ##Drop columns

# COMMAND ----------

# null values per column

# https://datatofish.com/count-nan-pandas-dataframe/

for col in df2.columns:
    count = df2[col].isnull().sum()
    print(col,df2[col].dtype,count)

# COMMAND ----------

# drop columns with too many null values > ~28,000

df3 = df2.drop(['auth_num','lit_ref','to_mfr','lot_num','exp_dt','nda_num','drug_rec_act','dur','dur_cod'], axis = 1)

display(df3)

# COMMAND ----------

# inspect remaining missing values in data

import missingno as msno

msno.matrix(df3)

# COMMAND ----------

df3.shape

# COMMAND ----------

# drop additional irrelevant columns that have too many missing values

df4 = df3.drop(['age_grp','cum_dose_chr','cum_dose_unit'], axis = 1)

display(df4)

# COMMAND ----------

# MAGIC %md ##Drop rows

# COMMAND ----------

# drop rows where null - age, sex, weight

# https://www.journaldev.com/33492/pandas-dropna-drop-null-na-values-from-dataframe

#df4 = df4.dropna(subset=['age','sex','wt','route','dose_amt','dechal','rechal'], axis = 0)
df4 = df4.dropna(subset=['age','sex','wt','route','dose_amt'], axis = 0)
#df4 = df4.dropna(subset=['age','sex','wt','route'], axis = 0)

# COMMAND ----------

df4.shape

# COMMAND ----------

# drop rows where age_cod <> 'YR'

# https://www.statology.org/pandas-drop-rows-with-value

df4 = df4[df4.age_cod == 'YR']

display(df4)

# COMMAND ----------

df4.shape

# COMMAND ----------

# inspect remaining missing values in data

import missingno as msno

msno.matrix(df4)

# COMMAND ----------

# spot inspect the data

# results show the need to consolidate the PT terms

df4[df4['caseid'] == 17639954].head(5)

# COMMAND ----------

# MAGIC %md ##New/recode features

# COMMAND ----------

df4.dtypes

# COMMAND ----------

# MAGIC %md ###Weight in lbs

# COMMAND ----------

# new feature - weight in lbs

# insert new columns next to 'wt' column 
df4.insert(loc = 18, 
          column = 'wt_in_lbs', 
          value = 0)

for index, row in df4.iterrows():
  df4['wt_in_lbs'] = df4['wt']*2.20462262

display(df4)

# COMMAND ----------

# MAGIC %md ###Preferred terms

# COMMAND ----------

# new feature - consolidate preferred terms

# insert new columns next to 'pt' column 
df4.insert(loc = 40, 
          column = 'pt_consolidated', 
          value = 0)

display(df4)

# COMMAND ----------

# MAGIC %md ###Dechallenge

# COMMAND ----------

# recode feature - Dechallenge (if reaction abated when drug therapy was stopped)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html

df4['dechal'].value_counts(dropna = False)

# COMMAND ----------

# consolidate all records with 'U' (Unknown), 'D' (Does Not Apply), and NaN to a single category 'U'

# https://www.geeksforgeeks.org/replace-all-the-nan-values-with-zeros-in-a-column-of-a-pandas-dataframe

df4['dechal'] = df4['dechal'].replace('D','U')
df4['dechal'] = df4['dechal'].replace(np.nan,'U')

# COMMAND ----------

df4['dechal'].value_counts(dropna = False)

# COMMAND ----------

# MAGIC %md ###Rechallenge

# COMMAND ----------

df4['rechal'].value_counts(dropna = False)

# COMMAND ----------

# consolidate all records with 'U' (Unknown), 'D' (Does Not Apply), and NaN to a single category 'U'

df4['rechal'] = df4['rechal'].replace('D','U')
df4['rechal'] = df4['dechal'].replace(np.nan,'U')

# COMMAND ----------

df4['rechal'].value_counts(dropna = False)

# COMMAND ----------

df4.shape

# COMMAND ----------

# MAGIC %md ##Export for AML

# COMMAND ----------

df4.shape

# COMMAND ----------

#2022-02-03 Modify dataset to drop columns for autoML training

df4 = df4.select_dtypes(exclude='object') \
                            .drop(['primaryid','caseid','caseversion','event_dt','mfr_dt','init_fda_dt','fda_dt','wt', \
                                    'rept_dt','last_case_version','val_vbm','start_dt','end_dt','drug_seq','dsg_drug_seq'], axis=1)

# COMMAND ----------

df4.dtypes

# COMMAND ----------

# export for Azure ML autoML

#df4.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess2_5589.csv', index=False)
df4.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess2_5429.csv', index=False)

# COMMAND ----------

df4.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess2_5429_removedCols.csv', index=False)

# COMMAND ----------

# MAGIC %md #Build a baseline model

# COMMAND ----------

df4.select_dtypes(exclude='object').dtypes

# COMMAND ----------

# curate feature set

df5 = df4.copy() 

# df6 = df5.select_dtypes(exclude='object').drop(['mfr_dt','wt','start_dt','end_dt'], axis=1)

df5 = df4.select_dtypes(exclude='object') \
                            .drop(['primaryid','caseid','caseversion','event_dt','mfr_dt','init_fda_dt','fda_dt','wt', \
                                    'rept_dt','last_case_version','val_vbm','start_dt','end_dt','drug_seq','dsg_drug_seq'], axis=1)

# COMMAND ----------

display(df5)

# COMMAND ----------

# X = input
X = df5.drop("outc_cod_DE" ,axis= 1)

# y = output
y = df5['outc_cod_DE']

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

# MAGIC %md ##Analyze data distribution

# COMMAND ----------

# MAGIC %md In order to prepare the data for machine learning tasks, we need to characterize the location and variability of the data.  A further characterization of the data includes data distribution, skewness and kurtosis.
# MAGIC 
# MAGIC Skewness - What is the shape of the distribution?  
# MAGIC 
# MAGIC Kurtosis - What is the measure of thickness or heaviness of the distribution?  
# MAGIC 
# MAGIC https://tekmarathon.com/2015/11/13/importance-of-data-distribution-in-training-machine-learning-models/

# COMMAND ----------

# which features are numeric data type

df4.select_dtypes(exclude='object').dtypes

# COMMAND ----------

# data distribution of numeric features

#https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3

df4.hist(figsize=(12, 12))

# COMMAND ----------

# MAGIC %md ##Correct skewed data

# COMMAND ----------

# MAGIC %md ###Sample skew & kurtosis

# COMMAND ----------

# visualize skew for a sample feature

# https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45
# https://www.analyticsvidhya.com/blog/2021/05/shape-of-data-skewness-and-kurtosis/
# https://opendatascience.com/transforming-skewed-data-for-machine-learning/

sns.distplot(df4['dose_amt'])

# COMMAND ----------

# calculate skew value

# https://www.geeksforgeeks.org/scipy-stats-skew-python/
# skewness > 0 : more weight in the left tail of the distribution.  

# https://medium.com/@TheDataGyan/day-8-data-transformation-skewness-normalization-and-much-more-4c144d370e55
# If skewness value lies above +1 or below -1, data is highly skewed. If it lies between +0.5 to -0.5, it is moderately skewed. If the value is 0, then the data is symmetric

# https://vivekrai1011.medium.com/skewness-and-kurtosis-in-machine-learning-c19f79e2d7a5
# If the peak of the distribution is in right side that means our data is negatively skewed and most of the people reported with AEs weigh more than the average.

df4['wt_in_lbs'].skew()

# COMMAND ----------

# calculate kurtosis value

df4['wt_in_lbs'].kurtosis() # platykurtic distribution (low degree of peakedness)

# COMMAND ----------

# MAGIC %md ###Log Transformation

# COMMAND ----------

# convert dataframe columns to list

# https://datatofish.com/convert-pandas-dataframe-to-list

num_col = df4.select_dtypes(exclude='object') \
              .columns.drop(['outc_cod_DE']) \
              .values.tolist()

print (num_col)

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
    skewval = df4[i].skew()
    skew_before.append(skewval)
    
    kurtval = df4[i].kurtosis()
    kurt_before.append(kurtval)
    
    sdval = df4[i].std()
    standard_deviation_before.append(sdval)
    
    if (abs(skewval) >1) & (abs(kurtval) >3):
        log_transform_needed.append('Yes')
        
        if len(df4[df4[i] == 0])/len(df4) <=0.02:
            log_type.append('log')
            skewvalnew = np.log(pd.DataFrame(df4[df4[i] > 0])[i]).skew()
            skew_after.append(skewvalnew)
            
            kurtvalnew = np.log(pd.DataFrame(df4[df4[i] > 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)
            
            sdvalnew = np.log(pd.DataFrame(df4[df4[i] > 0])[i]).std()
            standard_deviation_after.append(sdvalnew)
            
        else:
            log_type.append('log1p')
            skewvalnew = np.log1p(pd.DataFrame(df4[df4[i] >= 0])[i]).skew()
            skew_after.append(skewvalnew)
        
            kurtvalnew = np.log1p(pd.DataFrame(df4[df4[i] >= 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)
            
            sdvalnew = np.log1p(pd.DataFrame(df4[df4[i] >= 0])[i]).std()
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

for i in range(len(statdataframe)):
    if statdataframe['log_transform_needed'][i] == 'Yes':
        colname = str(statdataframe['numeric_column'][i])
        
        if statdataframe['log_type'][i] == 'log':
            df5 = df4[df4[colname] > 0]
            df5[colname + "_log"] = np.log(df5[colname])
            
        elif statdataframe['log_type'][i] == 'log1p':
            df5 = df4[df[colname] >= 0]
            df5[colname + "_log1p"] = np.log1p(df4[colname])    

# COMMAND ----------

# drop original columns that needed log transformation and use log1p counterparts

#df5 = df5.drop(['drug_seq','cum_dose_chr','dose_amt','dsg_drug_seq'], axis = 1)

# COMMAND ----------

# update list with numeric features (after log transformation)

#numerics = list(set(list(df5._get_numeric_data().columns))- {'outc_cod_DE'})

#numerics

# COMMAND ----------

# https://github.com/datamadness/Automatic-skewness-transformation-for-Pandas-DataFrame

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:42:46 2019
@author: DATAmadness
"""

##################################################
# A function that will accept a pandas dataframe
# and auto-transforms columns that exceeds threshold value
#  -  Offers choice between boxcox or log / exponential transformation
#  -  Automatically handles negative values
#  -  Auto recognizes positive /negative skewness

# Further documentation available here:
# https://datamadness.github.io/Skewness_Auto_Transform


import seaborn as sns
import numpy as np
import math
import scipy.stats as ss
import matplotlib.pyplot as plt

def skew_autotransform(DF, include = None, exclude = None, plot = False, threshold = 1, exp = False):
    
    #Get list of column names that should be processed based on input parameters
    if include is None and exclude is None:
        colnames = DF.columns.values
    elif include is not None:
        colnames = include
    elif exclude is not None:
        colnames = [item for item in list(DF.columns.values) if item not in exclude]
    else:
        print('No columns to process!')
    
    #Helper function that checks if all values are positive
    def make_positive(series):
        minimum = np.amin(series)
        #If minimum is negative, offset all values by a constant to move all values to positive teritory
        if minimum <= 0:
            series = series + abs(minimum) + 0.01
        return series
    
    
    #Go throug desired columns in DataFrame
    for col in colnames:
        #Get column skewness
        skew = DF[col].skew()
        transformed = True
        
        if plot:
            #Prep the plot of original data
            sns.set_style("darkgrid")
            sns.set_palette("Blues_r")
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            ax1 = sns.distplot(DF[col], ax=axes[0])
            ax1.set(xlabel='Original ' + col)
        
        #If skewness is larger than threshold and positively skewed; If yes, apply appropriate transformation
        if abs(skew) > threshold and skew > 0:
            skewType = 'positive'
            #Make sure all values are positive
            DF[col] = make_positive(DF[col])
            
            if exp:
               #Apply log transformation 
               DF[col] = DF[col].apply(math.log)
            else:
                #Apply boxcox transformation
                DF[col] = ss.boxcox(DF[col])[0]
            skew_new = DF[col].skew()
         
        elif abs(skew) > threshold and skew < 0:
            skewType = 'negative'
            #Make sure all values are positive
            DF[col] = make_positive(DF[col])
            
            if exp:
               #Apply exp transformation 
               DF[col] = DF[col].pow(10)
            else:
                #Apply boxcox transformation
                DF[col] = ss.boxcox(DF[col])[0]
            skew_new = DF[col].skew()
        
        else:
            #Flag if no transformation was performed
            transformed = False
            skew_new = skew
        
        #Compare before and after if plot is True
        if plot:
            print('\n ------------------------------------------------------')     
            if transformed:
                print('\n %r had %r skewness of %2.2f' %(col, skewType, skew))
                print('\n Transformation yielded skewness of %2.2f' %(skew_new))
                sns.set_palette("Paired")
                ax2 = sns.distplot(DF[col], ax=axes[1], color = 'r')
                ax2.set(xlabel='Transformed ' + col)
                plt.show()
            else:
                print('\n NO TRANSFORMATION APPLIED FOR %r . Skewness = %2.2f' %(col, skew))
                ax2 = sns.distplot(DF[col], ax=axes[1])
                ax2.set(xlabel='NO TRANSFORM ' + col)
                plt.show()
                

    return DF

# COMMAND ----------

"""
Created on Sat Feb 23 17:10:57 2019
@author: DATAmadness
"""

############################################
###### TEST for skew autotransform##########

import pandas as pd
import numpy as np
#from skew_autotransform import skew_autotransform

#Import test dataset - Boston huosing data
from sklearn.datasets import load_boston

#exampleDF = pd.DataFrame(load_boston()['data'], columns = load_boston()['feature_names'].tolist())

exampleDF = pd.DataFrame(df4, columns = num_col)


transformedDF = skew_autotransform(exampleDF.copy(deep=True), plot = True, 
                                   exp = True, threshold = 0.7, exclude = ['B','LSTAT'])

print('Original average skewness value was %2.2f' %(np.mean(abs(exampleDF.skew()))))
print('Average skewness after transformation is %2.2f' %(np.mean(abs(transformedDF.skew()))))

# COMMAND ----------

# MAGIC %md ##Standardize data  
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

# MAGIC %md ##Remove outliers

# COMMAND ----------

df5.display(5)

# COMMAND ----------

# remove outliers 

df5 = df5[(np.abs(sp.stats.zscore(df5[numerics])) < 3).all(axis=1)]

# COMMAND ----------

display(df5)

# COMMAND ----------

df5.shape

# COMMAND ----------

import seaborn as sns

sns.boxplot(x=df4['wt_in_lbs'])

# COMMAND ----------

# use z-score
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df4))
print(z)

# COMMAND ----------

# MAGIC %md ##Convert categorical features

# COMMAND ----------

# MAGIC %md Using categorical data in training machine learning models
# MAGIC 
# MAGIC One of the major problems with machine learning is that a lot of algorithms cannot work directly with categorical data. Categorical data [1] are variables that can take on one of a limited number of possible values. Some examples are:
# MAGIC 
# MAGIC - The sex of a person: female or male.
# MAGIC - The airline travel class: First Class, Business Class, and Economy Class.
# MAGIC - The computer vendor: Lenovo, HP, Dell, Apple, Acer, Asus, and Others.
# MAGIC 
# MAGIC Therefore, we need a way to convert categorical data into a numerical form and our machine learning algorithm can take in that as input.
# MAGIC 
# MAGIC https://towardsdatascience.com/what-is-one-hot-encoding-and-how-to-use-pandas-get-dummies-function-922eb9bd4970

# COMMAND ----------

# identify columns that are categorical (no intrinsic ordering to the categories) and convert to numerical

# https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-numerical-variables/

df4.select_dtypes(include='object').head(5).T

# COMMAND ----------

# convert object features to numerical

# https://towardsdatascience.com/encoding-categorical-features-21a2651a065c

# use pandas dummies method
df5 = pd.get_dummies(df4)

df5.head(5)

# COMMAND ----------

# convert select categorical features to numerical as these will be useful features for modeling

#df_converted = pd.get_dummies(df4, columns=['mfr_sndr','sex','occp_cod','reporter_country', 
#                                    'occr_country','role_cod','drugname','prod_ai','route','dechal','rechal','dose_freq'], drop_first = True)

df_converted = pd.get_dummies(df4, columns=['sex', 
                                 'occr_country','role_cod','drugname','prod_ai','route','dechal','rechal','dose_freq'])


df_converted.head(2)

# COMMAND ----------

df_converted.shape

# COMMAND ----------

# save data to ADLS Gen2

#df_converted.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess3_5589.csv', index=False)
df_converted.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess3_5429.csv', index=False)
