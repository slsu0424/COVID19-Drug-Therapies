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

# convert to pandas

df1 = df.toPandas()

# COMMAND ----------

# new feature - outcome code
# add as a new column

df1.insert(loc = 47, 
          column = 'outc_cod_DE', 
          value = 0)

display(df1)

# COMMAND ----------

# target will be binary classification - did a patient die?

df1['outc_cod_DE'] = np.where(df1['outc_cod']=='DE',1,0)

# COMMAND ----------

df1['outc_cod_DE'].value_counts()

# COMMAND ----------

df1.shape

# COMMAND ----------

# export for Azure ML autoML

df1.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess2a.csv', index=False)

# COMMAND ----------

# MAGIC %md ##Drop columns

# COMMAND ----------

# null values per column

# https://datatofish.com/count-nan-pandas-dataframe/

for col in df1.columns:
    count = df1[col].isnull().sum()
    print(col,df1[col].dtype,count)
    
#display(df4.sort_values(by='age', ascending=False))

# COMMAND ----------

# drop columns with too many null values > ~28,000

df2 = df1.drop(['auth_num','lit_ref','to_mfr','lot_num','exp_dt','nda_num','drug_rec_act','dur','dur_cod'], axis = 1)

#drop any column that has null values
#df2 = df1.dropna(axis=1)

display(df2)

# COMMAND ----------

# inspect remaining missing values in data

import missingno as msno

msno.matrix(df2)

# COMMAND ----------

df2.shape
#df3.shape

# COMMAND ----------

# drop additional irrelevant columns that have too many missing values

df3 = df2.drop(['age_grp','cum_dose_chr','cum_dose_unit'], axis = 1)

#drop any column that has null values
#df2 = df1.dropna(axis=1)

display(df3)

# COMMAND ----------

# MAGIC %md ##Drop rows

# COMMAND ----------

# drop rows where null - age, sex, weight

# https://www.journaldev.com/33492/pandas-dropna-drop-null-na-values-from-dataframe

#df3 = df3.dropna(subset=['age','sex','wt','route','dose_amt','dechal','rechal'], axis = 0)
df3 = df3.dropna(subset=['age','sex','wt','route','dose_amt'], axis = 0)
#df3 = df3.dropna(subset=['age','sex','wt'], axis = 0)

#display(df3)

# COMMAND ----------

# drop rows where age_cod <> 'YR'

# https://www.statology.org/pandas-drop-rows-with-value

df3 = df3[df3.age_cod == 'YR']

display(df3)

# COMMAND ----------

df3.shape

# COMMAND ----------

# inspect remaining missing values in data

import missingno as msno

msno.matrix(df3)

# COMMAND ----------

# spot inspect the data

# results show the need to consolidate the PT terms

df3[df3['caseid'] == 17639954].head(5)

# COMMAND ----------

# MAGIC %md ##New/recode features

# COMMAND ----------

df3.dtypes

# COMMAND ----------

# MAGIC %md ###Weight in lbs

# COMMAND ----------

# new feature - weight in lbs

# insert new columns next to 'wt' column 
df3.insert(loc = 18, 
          column = 'wt_in_lbs', 
          value = 0)

for index, row in df3.iterrows():
  df3['wt_in_lbs'] = df3['wt']*2.20462262

display(df3)

# COMMAND ----------

# MAGIC %md ###Preferred terms

# COMMAND ----------

# new feature - consolidate preferred terms

# insert new columns next to 'pt' column 
df3.insert(loc = 40, 
          column = 'pt_consolidated', 
          value = 0)

display(df3)

# COMMAND ----------

# MAGIC %md ###Dechallenge

# COMMAND ----------

# recode feature - Dechallenge (if reaction abated when drug therapy was stopped)

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html

df3['dechal'].value_counts(dropna = False)

# COMMAND ----------

#df3.insert(loc = 18, 
#          column = 'wt_in_lbs', 
#          value = 0)

#for index, row in df3.iterrows():
#  df3['wt_in_lbs'] = df3['wt']*2.20462262

#display(df3)

# COMMAND ----------

# consolidate all records with 'U' (Unknown), 'D' (Does Not Apply), and NaN to a single category 'U'

# https://www.geeksforgeeks.org/replace-all-the-nan-values-with-zeros-in-a-column-of-a-pandas-dataframe

df3['dechal'] = df3['dechal'].replace('D','U')
df3['dechal'] = df3['dechal'].replace(np.nan,'U')

# COMMAND ----------

df3['dechal'].value_counts(dropna = False)

# COMMAND ----------

# MAGIC %md ###Rechallenge

# COMMAND ----------

df3['rechal'].value_counts(dropna = False)

# COMMAND ----------

# consolidate all records with 'U' (Unknown), 'D' (Does Not Apply), and NaN to a single category 'U'

df3['rechal'] = df3['rechal'].replace('D','U')
df3['rechal'] = df3['dechal'].replace(np.nan,'U')

# COMMAND ----------

df3['rechal'].value_counts(dropna = False)

# COMMAND ----------

df3.shape

# COMMAND ----------

df3['route'].value_counts(dropna = False)

# COMMAND ----------

df3['dose_amt'].value_counts(dropna = False)

# COMMAND ----------

# MAGIC %md ##Create class labels

# COMMAND ----------

# outcome values

df3['outc_cod'].value_counts()

# COMMAND ----------

# new feature - outcome code
# add as a new column

#df3.insert(loc = 42, 
#          column = 'outc_cod_DE', 
#          value = 0)

#display(df3)

# COMMAND ----------

# target will be binary classification - did a patient die?

#df3['outc_cod_DE'] = np.where(df3['outc_cod']=='DE',1,0)

# COMMAND ----------

#df3['outc_cod_DE'].value_counts()

# COMMAND ----------

# MAGIC %md ###Remove dup caseid

# COMMAND ----------

# return all record where outcome of death = 1

display(df3.loc[df3['outc_cod_DE'] == 1])

# COMMAND ----------

# duplicate cases with multiple outcomes (including death)

# how to treat 2 different reactions (pt) that resulted in death?
# ignore pt field and take 1st record where outc_code_DE = 1

df3[df3['caseid'] == 18322071].head(5)

# COMMAND ----------

# collapse multiple caseid (FAERS case)

# where there are multiple AEs for the same caseid, collapse these records by taking the max record where outc_code = DE (1)

#https://stackoverflow.com/questions/49343860/how-to-drop-duplicate-values-based-in-specific-columns-using-pandas

df4 = df3.sort_values(by=['outc_cod_DE'], ascending = False) \
        .drop_duplicates(subset=['caseid'], keep = 'first')

# COMMAND ----------

# https://learningactors.com/how-to-use-sql-with-pandas/

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

q = """SELECT * \
      FROM df3 where caseid = 18255075;"""

#q = """SELECT *, MAX(outc_cod_DE) OVER (PARTITION BY caseid) as max_outc_cod_DE \
#         FROM df3 \
#         WHERE caseid = '18255075';"""

#q = """
#    SELECT * FROM \
#      (SELECT *, \
#         MAX(outc_cod_DE) OVER (PARTITION BY caseid) as max_outc_cod_DE \
#         FROM df3 \
#         WHERE caseid = '18255075') \
#      WHERE outc_cod_DE = max_outc_cod_DE;"""

#display(pysqldf(q))

# COMMAND ----------

# re-inspect same case

df4[df4['caseid'] == 18322071].head(5)

# COMMAND ----------

# how many records after removing dups
df4.shape

# COMMAND ----------

# count distinct cases

# https://datascienceparichay.com/article/pandas-count-of-unique-values-in-each-column/#:~:text=By%20default%2C%20the%20pandas%20dataframe%20nunique%20%28%29%20function,the%20count%20of%20distinct%20values%20in%20each%20column

print(df4.nunique())

#from pyspark.sql.functions import countDistinct

#df4.select(countDistinct("caseid")).show()

# COMMAND ----------

# MAGIC %md ###Are the classes balanced?

# COMMAND ----------

# value counts now

df4['outc_cod_DE'].value_counts()

# COMMAND ----------

# check label ratios
# https://dataaspirant.com/handle-imbalanced-data-machine-learning/

sns.countplot(x='outc_cod_DE', data=df4) # data already looks wildly imbalanced but let us continue

# COMMAND ----------

# MAGIC %md ##Export for AML

# COMMAND ----------

df4.shape

# COMMAND ----------

# export for Azure ML autoML

df4.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess2_5589.csv', index=False)

# COMMAND ----------

# MAGIC %md #Build baseline

# COMMAND ----------

# MAGIC %md ##Export for AML

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

df_converted = pd.get_dummies(df4, columns=['mfr_sndr','sex','occp_cod','reporter_country', 
                                    'occr_country','role_cod','drugname','prod_ai','route','dechal','rechal','dose_freq'])


df_converted.head(2)

# COMMAND ----------

df_converted.shape

# COMMAND ----------

# save data to ADLS Gen2

df_converted.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess3_5589.csv', index=False)
