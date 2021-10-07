# Databricks notebook source
# MAGIC %md #Part 2: Preprocessing & Feature Engineering

# COMMAND ----------

# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml
# https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114

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

# MAGIC %md #Reload data

# COMMAND ----------

df = spark.read.csv("/mnt/adls/FAERS_CSteroid_preprocess1.csv", header="true", nullValue = "NA", inferSchema="true")

display(df)

# COMMAND ----------

# how many rows, columns

print((df.count(), len(df.columns)))

# COMMAND ----------

# MAGIC %md #Exploratory Data Analysis

# COMMAND ----------

# convert to pandas

df1 = df.toPandas()

# COMMAND ----------

# data distribution of numerical variables

# https://thecleverprogrammer.com/2021/02/05/what-is-exploratory-data-analysis-eda/

# https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3

df1.hist(figsize=(12, 12))

# COMMAND ----------

# run dataframe statistics

df1.describe()

# COMMAND ----------

# detect outliers

# https://www.kaggle.com/agrawaladitya/step-by-step-data-preprocessing-eda
# https://www.machinelearningplus.com/plots/python-boxplot

# select numerical variables of interest
num_cols = ['age','wt','drug_seq','cum_dose_chr','dose_amt','dsg_drug_seq','dur']

plt.figure(figsize=(18,9))
df1[num_cols].boxplot()
plt.title("Numerical variables in the Corticosteroid dataset", fontsize=20)
plt.show()

# COMMAND ----------

# MAGIC %md #Preprocess data

# COMMAND ----------

# MAGIC %md ##Create target variable

# COMMAND ----------

# outcome values

df1['outc_cod'].value_counts()

# COMMAND ----------

# new column - outcome code = DE

df1.insert(loc = 52, 
          column = 'outc_cod_DE', 
          value = 0)

display(df1)

# COMMAND ----------

# target will be binary classification - did a patient die?

df1['outc_cod_DE'] = np.where(df1['outc_cod']=='DE',1,0)

# COMMAND ----------

# outcome values = DE only

df1['outc_cod_DE'].value_counts()

# COMMAND ----------

# merge dup caseid where outc_cod = DE is present

# return some records where outcome of death = 1

df1.loc[df1['outc_cod_DE'] == 1].head(5)

# COMMAND ----------

# collapse multiple caseid (FAERS case)

# where there are multiple AEs for the same caseid, collapse these records by taking the max record where outc_code = DE (1)

#https://stackoverflow.com/questions/49343860/how-to-drop-duplicate-values-based-in-specific-columns-using-pandas

df2 = df1.sort_values(by=['outc_cod_DE'], ascending = False) \
        .drop_duplicates(subset=['caseid'], keep = 'first')

# COMMAND ----------

# re-inspect same case

df2[df2['caseid'] == 18322071].head(5)

# COMMAND ----------

# how many records after removing dups

df2.shape

# COMMAND ----------

# count distinct cases

# https://datascienceparichay.com/article/pandas-count-of-unique-values-in-each-column

print(df2.nunique())

# COMMAND ----------

# value counts now

df2['outc_cod_DE'].value_counts()

# COMMAND ----------

# export for Azure ML autoML

df2.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess2.csv', index=False)

# COMMAND ----------

# MAGIC %md ##Drop NULL columns

# COMMAND ----------

# display all null values per column

# https://www.analyticsvidhya.com/blog/2021/10/a-beginners-guide-to-feature-engineering-everything-you-need-to-know/

df2.isnull().sum()

# COMMAND ----------

# drop columns with > 90% missing values

threshold = 0.8
df3 = df2[df2.columns[df2.isnull().mean() < threshold]]
df3.columns

# COMMAND ----------

df3.shape

# COMMAND ----------

# inspect remaining missing values in data

import missingno as msno

msno.matrix(df3)

# COMMAND ----------

# MAGIC %md ##Impute missing values

# COMMAND ----------

# MAGIC %md ###Numerical

# COMMAND ----------

# check null values in numerical variables 

df3.select_dtypes(exclude='object').isnull().sum()

# COMMAND ----------

# we are interested to impute for age, wt, dose_amt

# use mean (not too many outliers) or median (skewed distribution).  Median will be used due to skewed distributions.

# https://machinelearningbites.com/missing-values-imputation-strategies/
# https://vitalflux.com/imputing-missing-data-sklearn-simpleimputer/
# https://www.shanelynn.ie/pandas-iloc-loc-select-rows-and-columns-dataframe/#1-pandas-iloc-data-selection
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
# https://datascientyst.com/reshape-pandas-series-into-2d-array/

from sklearn.impute import SimpleImputer

df4 = df3.copy() 

imputer = SimpleImputer(missing_values=np.nan, strategy= 'median')

df4.age = imputer.fit_transform(df4['age'].values.reshape(-1,1))
df4.wt = imputer.fit_transform(df4['wt'].values.reshape(-1,1))
df4.dose_amt = imputer.fit_transform(df4['dose_amt'].values.reshape(-1,1))

display(df4)

# COMMAND ----------

# check nulls again

df4.select_dtypes(exclude='object').isnull().sum()

# COMMAND ----------

# MAGIC %md ###Categorical

# COMMAND ----------

# check null values in categorical variables 

df4.select_dtypes(include='object').isnull().sum()

# COMMAND ----------

# we are interested to impute for sex, route, dechal, dose_freq

df5 = df4.copy()

imputer = SimpleImputer(missing_values=None, strategy= 'most_frequent')
df5.sex = imputer.fit_transform(df5['sex'].values.reshape(-1,1))
df5.route = imputer.fit_transform(df5['route'].values.reshape(-1,1))
df5.dechal = imputer.fit_transform(df5['dechal'].values.reshape(-1,1))
df5.dose_freq = imputer.fit_transform(df5['dose_freq'].values.reshape(-1,1))

display(df4)

# COMMAND ----------

# check nulls again

df5.select_dtypes(exclude='object').isnull().sum()

# COMMAND ----------

# inspect remaining missing values in data

import missingno as msno

msno.matrix(df5)

# COMMAND ----------

# MAGIC %md ##Drop NULL rows

# COMMAND ----------

#df5.isnull()

# COMMAND ----------

# drop rows where age_cod <> 'YR'

# https://www.statology.org/pandas-drop-rows-with-value

df3 = df3[df3.age_cod == 'YR']

display(df3)

# COMMAND ----------

# MAGIC %md #Feature Engineering

# COMMAND ----------

# MAGIC %md ##Encode categorical variables

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

# convert categorical features to numerical

# https://towardsdatascience.com/encoding-categorical-features-21a2651a065c

# use pandas dummies method
df5 = pd.get_dummies(df4)

df5.head(5)

# COMMAND ----------

# 2021-09-27 - Dropped reporter country as not relevant

# convert select categorical features to numerical as these will be useful features for modeling

#df_converted = pd.get_dummies(df4, columns=['mfr_sndr','sex','occp_cod','reporter_country', 
#                                    'occr_country','role_cod','drugname','prod_ai','route','dechal','rechal','dose_freq'], drop_first = True)

df_converted = pd.get_dummies(df4, columns=['mfr_sndr','sex','occp_cod', 
                                    'occr_country','role_cod','drugname','prod_ai','route','dechal','rechal','dose_freq'])

df_converted.head(2)

# COMMAND ----------

# MAGIC %md ##Recode features

# COMMAND ----------

# MAGIC %md ###Weight in lbs

# COMMAND ----------

# new feature - weight in lbs

# insert new columns next to 'wt' column 
df3.insert(loc = 16, 
          column = 'wt_in_lbs', 
          value = 0)

for index, row in df3.iterrows():
  df3['wt_in_lbs'] = df3['wt']*2.20462262

display(df3)

# COMMAND ----------

# MAGIC %md ##Scale Data

# COMMAND ----------

# MAGIC %md ###Correct skew & kurtosis

# COMMAND ----------

# MAGIC %md In order to prepare the data for machine learning tasks, we need to characterize the location and variability of the data.  A further characterization of the data includes data distribution, skewness and kurtosis.
# MAGIC 
# MAGIC Skewness - What is the shape of the distribution?  
# MAGIC 
# MAGIC Kurtosis - What is the measure of thickness or heaviness of the distribution?  
# MAGIC 
# MAGIC https://tekmarathon.com/2015/11/13/importance-of-data-distribution-in-training-machine-learning-models/

# COMMAND ----------

# visualize skew for a sample feature

# https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45
# https://www.analyticsvidhya.com/blog/2021/05/shape-of-data-skewness-and-kurtosis/
# https://opendatascience.com/transforming-skewed-data-for-machine-learning/

sns.distplot(df4['age'])

# COMMAND ----------

# calculate skew value

# https://www.geeksforgeeks.org/scipy-stats-skew-python/
# skewness > 0 : more weight in the left tail of the distribution.  

# https://medium.com/@TheDataGyan/day-8-data-transformation-skewness-normalization-and-much-more-4c144d370e55
# If skewness value lies above +1 or below -1, data is highly skewed. If it lies between +0.5 to -0.5, it is moderately skewed. If the value is 0, then the data is symmetric

# https://vivekrai1011.medium.com/skewness-and-kurtosis-in-machine-learning-c19f79e2d7a5
# If the peak of the distribution is in right side that means our data is negatively skewed and most of the people reported with AEs weigh more than the average.

df4['dose_amt'].skew()

# COMMAND ----------

# calculate kurtosis value

df4['dose_amt'].kurtosis() # platykurtic distribution (low degree of peakedness)

# COMMAND ----------

# MAGIC %md ###Log Transformation

# COMMAND ----------

# convert dataframe columns to list

# https://datatofish.com/convert-pandas-dataframe-to-list

num_col = df4.select_dtypes(exclude='object') \
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
    skewval = df4[i].skew()
    skew_before.append(skewval)
    
    kurtval = df4[i].kurtosis()
    kurt_before.append(kurtval)
    
    sdval = df4[i].std()
    standard_deviation_before.append(sdval)
    
    # https://quick-adviser.com/what-does-the-kurtosis-value-tell-us
    if (abs(skewval) >2) & (abs(kurtval) >2):
        log_transform_needed.append('Yes')
        
        # are there any features that have values of 0 (no predictive power)?
        if len(df4[df4[i] == 0])/len(df4) <= 0.02:
            log_type.append('log')
            skewvalnew = np.log(pd.DataFrame(df4[df4[i] > 0])[i]).skew()
            skew_after.append(skewvalnew)
            
            kurtvalnew = np.log(pd.DataFrame(df4[df4[i] > 0])[i]).kurtosis()
            kurt_after.append(kurtvalnew)
            
            sdvalnew = np.log(pd.DataFrame(df4[df4[i] > 0])[i]).std()
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

df_converted.shape

# COMMAND ----------

# save data to ADLS Gen2

df_converted.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess3.csv', index=False)

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

# null values per column

# https://datatofish.com/count-nan-pandas-dataframe/

#for col in df1.columns:
#    count = df1[col].isnull().sum()
#    print(col,df1[col].dtype,count)

# COMMAND ----------

# drop columns with too many null values > ~28,000

df2 = df1.drop(['auth_num','lit_ref','to_mfr','lot_num','exp_dt','nda_num','drug_rec_act','dur','dur_cod'], axis = 1)

#drop any column that has null values
#df2 = df1.dropna(axis=1)

display(df2)

# COMMAND ----------

# spot inspect the data

# results show the need to consolidate the PT terms

df3[df3['caseid'] == 17639954].head(5)

# COMMAND ----------

# drop rows where null - age, sex, weight

# https://www.journaldev.com/33492/pandas-dropna-drop-null-na-values-from-dataframe

#df3 = df3.dropna(subset=['age','sex','wt','route','dose_amt','dechal','rechal'], axis = 0)
#df3 = df3.dropna(subset=['age','sex','wt','route','dose_amt'], axis = 0)
#df3 = df3.dropna(subset=['age','sex','wt'], axis = 0)

#display(df3)

# COMMAND ----------

# check label ratios
# https://dataaspirant.com/handle-imbalanced-data-machine-learning/

#sns.countplot(x='outc_cod_DE', data=df4) 

# 0 is majority class
# 1 is minority class

# data already looks wildly imbalanced but let us continue

# COMMAND ----------

# drop additional irrelevant columns that have too many missing values

df6 = df5.drop(['dose_form'], axis = 1)

#drop any column that has null values
#df2 = df1.dropna(axis=1)

display(df6)
