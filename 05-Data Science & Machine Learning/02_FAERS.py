# Databricks notebook source
# MAGIC %md #Part 2: Exploratory Data Analysis & Preprocessing for AutoML

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

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md #Load data

# COMMAND ----------

df = spark.read.csv("/mnt/adls/FAERS_CSteroid_preprocess1.csv", header="true", nullValue = "NA", inferSchema="true")

display(df)

# COMMAND ----------

# how many rows, columns

print((df.count(), len(df.columns)))

# COMMAND ----------

# MAGIC %md #Explore data

# COMMAND ----------

# Exploratory Data Analysis (EDA)

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

# MAGIC %md #Preprocess Data

# COMMAND ----------

# MAGIC %md ##Drop NULL values

# COMMAND ----------

# MAGIC %md ###NULL columns

# COMMAND ----------

# get all column types

df1.dtypes

# COMMAND ----------

# sum of null values per column

# https://www.analyticsvidhya.com/blog/2021/10/a-beginners-guide-to-feature-engineering-everything-you-need-to-know/

df1.isnull().sum()

# COMMAND ----------

# drop columns with > 95% missing values

threshold = 0.95
df2 = df1[df1.columns[df1.isnull().mean() < threshold]]

df2.columns

# COMMAND ----------

df2.shape

# COMMAND ----------

# visually inspect remaining missing values in data

import missingno as msno

msno.matrix(df2)

# COMMAND ----------

# MAGIC %md ###NULL rows

# COMMAND ----------

# null values per numerical column

df2.select_dtypes(exclude='object').isnull().sum()

# COMMAND ----------

# drop rows with Nan Values in ALL columns of interest

df3 = df2.dropna(subset=['age', 'wt', 'dose_amt'],how='all')

#df4 = df3.dropna(subset=['age', 'wt', 'dose_amt'])

# COMMAND ----------

df3.shape

# COMMAND ----------

# MAGIC %md ##Create target variable

# COMMAND ----------

# outcome values

df3['outc_cod'].value_counts()

# COMMAND ----------

# new column - outcome code = DE

df3.insert(loc = 41, 
          column = 'outc_cod_DE', 
          value = 0)

#display(df3)

# COMMAND ----------

# target will be binary classification - did a patient die?

df3['outc_cod_DE'] = np.where(df3['outc_cod']=='DE',1,0)

# COMMAND ----------

# outcome values = DE only

df3['outc_cod_DE'].value_counts()

# COMMAND ----------

# MAGIC %md ###Merge dup caseid

# COMMAND ----------

# merge dup caseid where outc_cod = DE is present

# return some records where outcome of death = 1

df3.loc[df3['outc_cod_DE'] == 1].head(5)

# COMMAND ----------

# collapse multiple caseid (FAERS case)

# where there are multiple AEs for the same caseid, collapse these records by taking the max record where outc_code = DE (1)

#https://stackoverflow.com/questions/49343860/how-to-drop-duplicate-values-based-in-specific-columns-using-pandas

df4 = df3.sort_values(by=['outc_cod_DE'], ascending = False) \
        .drop_duplicates(subset=['caseid'], keep = 'first')

# COMMAND ----------

# re-inspect case

df4[df4['caseid'] == 18322071].head(5)

# COMMAND ----------

# how many records after removing dups

df4.shape

# COMMAND ----------

# count distinct cases

# https://datascienceparichay.com/article/pandas-count-of-unique-values-in-each-column

print(df4.nunique())

# COMMAND ----------

# new value counts

df4['outc_cod_DE'].value_counts()

# COMMAND ----------

# MAGIC %md ### Check for imbalance

# COMMAND ----------

# Are the target classes imbalanced?

# https://dataaspirant.com/handle-imbalanced-data-machine-learning/

# 0 is majority class
# 1 is minority class

sns.countplot(x='outc_cod_DE', data=df4) # data already looks wildly imbalanced but let us continue

# COMMAND ----------

# MAGIC %md ##Recode input variables

# COMMAND ----------

# MAGIC %md ###Convert columns

# COMMAND ----------

# age code

# DEC DECADE
# YR YEAR
# MON MONTH
# WK WEEK
# DY DAY
# HR HOUR

df4['age_cod'].value_counts(dropna = False) # https://www.statology.org/pandas-drop-rows-with-value

# COMMAND ----------

# age in years - insert new column next to 'age' column

df4.insert(loc = 15, 
  column = 'age_in_yrs', 
  value = '0')

# COMMAND ----------

# convert age to years

df4 = df4.copy(deep=True)

for index, row in df4.iterrows():
    if (row['age_cod'] == "YR"):
        df4.loc[index, 'age_in_yrs'] = df4.loc[index, 'age']
    elif (row['age_cod'] == "DEC"):
        df4.loc[index, 'age_in_yrs'] = df4.loc[index, 'age'] * 10
    elif (row['age_cod'] == "MON"):
        df4.loc[index, 'age_in_yrs'] = df4.loc[index, 'age'] / 12
    elif (row['age_cod'] == "WK"):
        df4.loc[index, 'age_in_yrs'] = df4.loc[index, 'age'] / 52
    elif (row['age_cod'] == "DY"):
        df4.loc[index, 'age_in_yrs'] = df4.loc[index, 'age'] / 365
    else:
        df4.loc[index, 'age_in_yrs'] = 0
    
# Test
df4[df4['age_cod'] == "DY"].head(5)

# COMMAND ----------

# weight code

df4['wt_cod'].value_counts()

# COMMAND ----------

# spot inspect

df4[df4['wt_cod'] == "LBS"].head(5)

# COMMAND ----------

# weight in lbs - insert new column next to 'wt' column

df4.insert(loc = 20, 
  column = 'wt_in_lbs', 
  value = 0)

# COMMAND ----------

# convert to lbs

for index, row in df4.iterrows():
    if (row['wt_cod'] == "KG"): # https://www.learndatasci.com/solutions/python-valueerror-truth-value-series-ambiguous-use-empty-bool-item-any-or-all/
        df4.loc[index, 'wt_in_lbs'] = df4.loc[index, 'wt'] * 2.20462262
    else:
        df4.loc[index, 'wt_in_lbs'] = df4.loc[index, 'wt']

df4.head(5)

# COMMAND ----------

# convert new columns to numeric

# https://stackoverflow.com/questions/21197774/assign-pandas-dataframe-column-dtypes

df4["age_in_yrs"] = pd.to_numeric(df4["age_in_yrs"])
df4["wt_in_lbs"] = pd.to_numeric(df4["wt_in_lbs"])    

# COMMAND ----------

# MAGIC %md ###Convert 0 to NULL

# COMMAND ----------

df4.select_dtypes(exclude='object').isin([0]).sum()

# COMMAND ----------

#df4 = df4.copy(deep=True)

df4['age_in_yrs'].replace(0, np.nan, inplace=True)
df4['wt_in_lbs'].replace(0, np.nan, inplace=True)
df4['dose_amt'].replace(0, np.nan, inplace=True)

# COMMAND ----------

df4.select_dtypes(exclude='object').isin([0]).sum()

# COMMAND ----------

# MAGIC %md ##Remove outliers

# COMMAND ----------

# detect outliers

# https://www.kaggle.com/agrawaladitya/step-by-step-data-preprocessing-eda
# https://www.machinelearningplus.com/plots/python-boxplot

# select numerical variables of interest
num_cols = ['age_in_yrs','wt_in_lbs','drug_seq','dose_amt']

plt.figure(figsize=(18,9))
df4[num_cols].boxplot()
plt.title("Numerical variables in the Corticosteroids dataset", fontsize=20)
plt.show()

# COMMAND ----------

import seaborn as sns

sns.boxplot(x=df4['dose_amt'])

# COMMAND ----------

sns.boxplot(x=df4['wt_in_lbs'])

# COMMAND ----------

df4['wt_in_lbs'].nlargest(n=10)

# COMMAND ----------

# https://www.cdc.gov/obesity/adult/defining.html

#df5 = df4[df4.wt_in_lbs != 1690.94554954]
#df5 = df4.drop(df4.index[df4['wt_in_lbs'] == '1124.357536'], inplace=True)

# COMMAND ----------

df5 = df4

df5.shape

# COMMAND ----------

# MAGIC %md ##Drop irrelevant variables

# COMMAND ----------

# drop columns that will not be used for training and inference

df5.dtypes

# COMMAND ----------

# drop columns

df6 = df5.drop(['primaryid', 'caseid', 'caseversion', 'i_f_code', \
                'event_dt', 'mfr_dt', 'init_fda_dt', 'fda_dt', \
                'rept_cod', 'auth_num', 'mfr_num', 'age', \
                'age_cod', 'age_grp', 'e_sub', 'wt', \
                'wt_cod', 'rept_dt', 'to_mfr', 'occp_cod', \
                'reporter_country', 'last_case_version', 'role_cod', 'prod_ai', 
                'val_vbm', 'dose_vbm', 'cum_dose_chr', 'cum_dose_unit', \
                'lot_num', 'nda_num', 'dose_unit', 'dose_form', \
                'dose_freq', 'drug_seq', 'dsg_drug_seq', 'pt', \
                'outc_cod', 'start_dt', 'end_dt'], axis=1)


# COMMAND ----------

# final columns to be used for training

df6.dtypes

# COMMAND ----------

df6.shape

# COMMAND ----------

# MAGIC %md #Build baseline classifier

# COMMAND ----------

# input
X = df6.drop('outc_cod_DE', axis= 1)

# output
y = df6['outc_cod_DE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# show size of each dataset (records, columns)
print("Dataset sizes: \nX_train", X_train.shape," \nX_test", X_test.shape, " \ny_train", y_train.shape, "\ny_test", y_test.shape)

data = {
    "train":{"X": X_train, "y": y_train},        
    "test":{"X": X_test, "y": y_test}
}

print ("Data contains", len(data['train']['X']), "training samples and",len(data['test']['X']), "test samples")

# COMMAND ----------

strategies = ['most_frequent', 'stratified', 'uniform', 'constant']

test_scores = []
for s in strategies:
	if s =='constant':
		dclf = DummyClassifier(strategy = s, random_state = 0, constant = 1)
	else:
		dclf = DummyClassifier(strategy = s, random_state = 0)
	dclf.fit(X_train, y_train)
	score = dclf.score(X_test, y_test)
	test_scores.append(score)
    
print(test_scores)

# COMMAND ----------

# view visually

# constant strategy (predicting minority class) --> most closely approximates F-1 measure.  Any model would have to do better than F-1 >= 0.16

ax = sns.stripplot(strategies, test_scores);
ax.set(xlabel ='Strategy', ylabel ='Test Score')
plt.show()

# COMMAND ----------

# MAGIC %md #Save to ADLS

# COMMAND ----------

# save data to ADLS Gen2

df6.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess2.csv', index=False)
