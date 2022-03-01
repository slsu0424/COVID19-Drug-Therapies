# Databricks notebook source
# unmount folders as needed 
dbutils.fs.unmount("/mnt/adls")

# COMMAND ----------

sp_secret = dbutils.secrets.get(scope = "secret01", key = "testsecret01")

configs = {"fs.azure.account.auth.type": "OAuth",
           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
           "fs.azure.account.oauth2.client.id": "029b4a8c-607f-42e7-9947-52d13d2bb4a9",
           "fs.azure.account.oauth2.client.secret": sp_secret,
           "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47/oauth2/token"}

dbutils.fs.mount(
  source = "abfss://sdudsynapsefilesystem@sdudsynapseadls2.dfs.core.windows.net/",
  mount_point = "/mnt/adls",
  extra_configs = configs)

# COMMAND ----------

display(dbutils.fs.ls("/mnt/adls"))

# COMMAND ----------

# MAGIC %md ##Load data

# COMMAND ----------

#Load dataset from DBFS into Spark dataframe

df_18 = spark.read.format('parquet').options(header='true', inferSchema='true').load("/mnt/adls/State_Drug_Utilization_Data_2018.parquet")

display(df_18.limit(10))
#df_18.show(5)

# COMMAND ----------

# read 2018 data
df_18 = spark.read.csv("/mnt/adls/State_Drug_Utilization_Data_2018.csv", header="true", nullValue = "NA", inferSchema="true")

display(df_18)

# COMMAND ----------

# read 2019 data
df_19 = spark.read.csv("/mnt/adls/State_Drug_Utilization_Data_2019.csv", header="true", nullValue = "NA", inferSchema="true")

display(df_19)

# COMMAND ----------

# read 2020 data
df_20 = spark.read.csv("/mnt/adls/State_Drug_Utilization_Data_2020.csv", header="true", nullValue = "NA", inferSchema="true")

display(df_20)

# COMMAND ----------

# read 2021 data
df_21 = spark.read.csv("/mnt/adls/State_Drug_Utilization_Data_2021.csv", header="true", nullValue = "NA", inferSchema="true")

display(df_21)

# COMMAND ----------

print((df_21.count(), len(df_21.columns)))

# COMMAND ----------

df_test = df_18.toPandas()

# COMMAND ----------

df_test.to_parquet('/dbfs/mnt/adls/FAERS_CSteroid_preprocess_2018test.parquet', index=False)

# COMMAND ----------

# https://www.tutorialkart.com/apache-spark/spark-append-concatenate-datasets-example/#:~:text=Append%20or%20Concatenate%20Datasets%20Spark%20provides%20union%20%28%29,on%20Datasets%20with%20the%20same%20number%20of%20columns

# https://bigdataprogrammers.com/merge-multiple-data-frames-in-spark/

df_all = df_18.union(df_19).union(df_20).union(df_21)

#display(df_all)

# COMMAND ----------

display(df_all)

# COMMAND ----------

# print schema
df_all.printSchema()

# COMMAND ----------

# rename columns

# https://www.datasciencemadesimple.com/rename-column-name-in-pyspark-single-multiple-column/

oldColumns = df_all.schema.names
newColumns = ["Utilization_Type", "State", "Labeler_Code", "Product_Code",
       "Package_Size", "Year", "Quarter", "Product_Name", "Supression Used",
       "Units_Reimbursed", "Number_of_Prescriptions",
       "Total_Amount_Reimbursed", "Medicaid_Amount_Reimbursed",
       "Non_Medicaid_Amount_Reimbursed", "Quarter_Begin", "Quarter_Begin_Date",
       "Latitude", "Longitude", "Location", "NDC"]

from functools import reduce

df1 = reduce(lambda df_all, idx: df_all.withColumnRenamed(oldColumns[idx], newColumns[idx]), range(len(oldColumns)), df_all)

# COMMAND ----------

# print new schema
df1.printSchema()

# COMMAND ----------

# how many rows, columns
print((df1.count(), len(df1.columns)))

# COMMAND ----------

# MAGIC %md ##Explore the data

# COMMAND ----------

# run statistics

# https://stackoverflow.com/questions/55938112/describe-a-dataframe-on-pyspark
# https://sparkbyexamples.com/spark/spark-show-display-dataframe-contents-in-table

display(df1.summary())

# COMMAND ----------

# create temp view for initial exploration/querying

# https://analyticshut.com/running-sql-queries-on-spark-dataframes/

df1.createOrReplaceTempView("SDUD2020")

spark.sql("select count(*) from SDUD2020").show() 

# COMMAND ----------

# All products

#https://intellipaat.com/blog/tutorial/spark-tutorial/pyspark-sql-cheat-sheet/

display(spark.sql("select * from SDUD2020 order by Product_Name asc"))

#display(spark.sql("select * from SDUD2020 where Product_Name LIKE ('%DEXAMETHA%') order by Number_of_Prescriptions desc"))

# COMMAND ----------

# Top # Rx by Drug - by Year

display(spark.sql("select Product_Name, Year, sum(Number_of_Prescriptions) \
                   from SDUD2020 \
                   group by Product_Name, Year \
                   order by sum(Number_of_Prescriptions) desc, Year desc, Product_Name asc"))

# COMMAND ----------

# Top 10 Rx by drug - 2020 only

display(spark.sql("select Product_Name, sum(Number_of_Prescriptions) \
                   from SDUD2020 \
                   where Year ='2020' \
                   and \
                   Product_Name <> 'UNKNOWN' \
                   group by Product_Name \
                   order by sum(Number_of_Prescriptions) desc \
                   limit 100"))

# COMMAND ----------

# 2018-2020 - Top # Rx by drug - Albuterol + Corticosteroids

display(spark.sql("select Product_Name, Year, Quarter, sum(Number_of_Prescriptions) \
                   from SDUD2020 \
                   where Product_Name LIKE ('%ALBUTEROL%') \
                   OR \
                   Product_Name LIKE ('%DEXAMETHA%') \
                   OR \
                   Product_Name LIKE ('%PREDNISON%') \
                   OR \
                   Product_Name LIKE ('%METHYLPRE%') \
                   group by Product_Name, Year, Quarter \
                   order by Year asc, Quarter asc, Product_Name asc, sum(Number_of_Prescriptions) desc"))

# COMMAND ----------

# 2020 only - Top Rx by drug - Albuterol + Corticosteroids

display(spark.sql("select Product_Name, Year, Quarter, sum(Number_of_Prescriptions) \
                   from SDUD2020 \
                   where Year = '2020' \
                   and (Product_Name = 'ALBUTEROL' \
                         OR \
                         Product_Name LIKE ('%DEXAMETHA%') \
                         OR \
                         Product_Name LIKE ('%PREDNISON%') \
                         OR \
                         Product_Name LIKE ('%METHYLPRED%')) \
                   group by Product_Name, Year, Quarter \
                   order by Year asc, Product_Name asc, sum(Number_of_Prescriptions) desc"))

# COMMAND ----------

# Top Rx by drug - Prednisone only (2018-2020)

display(spark.sql("select Product_Name, Year, Quarter, sum(Number_of_Prescriptions) \
                   from SDUD2020 \
                   where Product_Name LIKE ('%PREDNISON%') \
                   group by Product_Name, Year, Quarter \
                   order by Year asc, Quarter asc, Product_Name asc, sum(Number_of_Prescriptions) desc"))

# COMMAND ----------

# Most expensive Rx reimbursed by the govt (Medicaid)
# How much did state Medicaid programs pay for each drug?

display(spark.sql("select Product_Name, sum(Medicaid_Amount_Reimbursed) \
                   from SDUD2020 \
                   where Product_Name <> 'UNKNOWN' \
                   group by Product_Name \
                   order by sum(Medicaid_Amount_Reimbursed) desc"))

# COMMAND ----------

# Most expensive Rx reimbursed by the govt (Medicaid) - corticosteroids (2018-2020)
# How much did state Medicaid programs pay for each drug?

display(spark.sql("select Product_Name, Year, sum(Medicaid_Amount_Reimbursed) \
                   from SDUD2020 \
                   where Product_Name = 'ALBUTEROL' \
                   OR \
                   Product_Name LIKE ('%DEXAMETHA%') \
                   OR \
                   Product_Name LIKE ('%PREDNISON%') \
                   OR \
                   Product_Name LIKE ('%METHYLPRED%') \
                   group by Product_Name, Year \
                   order by Year asc, Product_Name asc, sum(Medicaid_Amount_Reimbursed) desc"))

# COMMAND ----------

# plot by state (2020 only)

display(spark.sql("select Product_Name, State, sum(Number_of_Prescriptions) \
                   from SDUD2020 \
                   where Year = '2020' \
                   and State <> 'XX' \
                   and (Product_Name = 'ALBUTEROL' \
                         OR \
                         Product_Name LIKE ('%DEXAMETHA%') \
                         OR \
                         Product_Name LIKE ('%PREDNISON%') \
                         OR \
                         Product_Name LIKE ('%METHYLPRED%')) \
                   group by Product_Name, State \
                   order by Product_Name asc, sum(Number_of_Prescriptions) desc"))

# COMMAND ----------

# MAGIC %md #Preprocess Data

# COMMAND ----------

# recode feature - consolidate Product_Name 

# https://www.datasciencemadesimple.com/sort-the-dataframe-in-pyspark-single-multiple-column/#:~:text=In%20order%20to%20sort%20the%20dataframe%20in%20pyspark,in%20pyspark%20by%20descending%20order%20or%20ascending%20order.

display(df1.orderBy('Product_Name', ascending=True))

# COMMAND ----------

# drop rows where Product_Name = 'null'

# https://www.geeksforgeeks.org/delete-rows-in-pyspark-dataframe-based-on-multiple-conditions/

df2 = df1.filter(df1.Product_Name != "null").orderBy('Product_Name', ascending=True)

display(df2)

# COMMAND ----------

# select distinct Product_Name

# https://www.datasciencemadesimple.com/distinct-value-of-a-column-in-pyspark/#:~:text=Distinct%20value%20of%20the%20column%20in%20pyspark%20is,value%20of%20those%20columns%20combined.%201%202%203
# https://amiradata.com/pyspark-distinct-value-of-a-column/

from pyspark.sql import functions

df2_distinct = df2.select("Product_Name").distinct().orderBy('Product_Name', ascending=True)

display(df2_distinct)

# COMMAND ----------

# check for similar Product_Names

#https://origin.geeksforgeeks.org/pyspark-filter-dataframe-based-on-multiple-conditions/

df_dex = df2_distinct.filter(df2_distinct.Product_Name.startswith('DEXAMET'))
df_pre = df2_distinct.filter(df2_distinct.Product_Name.startswith('PREDNIS'))
df_met = df2_distinct.filter(df2_distinct.Product_Name.startswith('METHYLP'))


display(df_dex)
display(df_pre)
display(df_met)

# COMMAND ----------

# consolidate Product_Name

# https://sparkbyexamples.com/pyspark/pyspark-when-otherwise/#:~:text=PySpark%20When%20Otherwise%20%E2%80%93%20when%20%28%29%20is%20a,WHEN%20cond1%20THEN%20result%20WHEN%20cond2%20THEN%20result

from pyspark.sql.functions import when

df3 = df2.withColumn('Product_Name', 
      when(df2.Product_Name == "DEXAMETHA", "DEXAMETHASONE") \
     .when(df2.Product_Name == "DEXAMETHAS", "DEXAMETHASONE") \
     .when(df2.Product_Name == "PREDNISON", "PREDNISONE") \
     .otherwise(df2.Product_Name))

# COMMAND ----------

df3_distinct = df3.select("Product_Name").distinct().orderBy('Product_Name', ascending=True)

# COMMAND ----------

df_dex1 = df3_distinct.filter(df3_distinct.Product_Name.startswith('DEXAMET'))

display(df_dex1)

# COMMAND ----------

df3.createOrReplaceTempView("SDUD2020_merge")

# COMMAND ----------

# Top # Rx by Drug - by Year

display(spark.sql("select Product_Name, Year, sum(Number_of_Prescriptions) \
                   from SDUD2020_merge \
                   group by Product_Name, Year \
                   order by sum(Number_of_Prescriptions) desc, Year desc, Product_Name asc"))

# COMMAND ----------

# Top 10 Rx by drug - 2020

display(spark.sql("select Product_Name, sum(Number_of_Prescriptions) \
                   from SDUD2020_merge \
                   where Year ='2020' \
                   and \
                   Product_Name <> 'UNKNOWN' \
                   group by Product_Name \
                   order by sum(Number_of_Prescriptions) desc \
                   limit 10"))

# COMMAND ----------

# 2018-2020 - Top # Rx by drug - Albuterol + Corticosteroids

display(spark.sql("select Product_Name, Year, Quarter, sum(Number_of_Prescriptions) \
                   from SDUD2020_merge \
                   where Product_Name LIKE ('%ALBUTEROL%') \
                   OR \
                   Product_Name LIKE ('%DEXAMETHA%') \
                   OR \
                   Product_Name LIKE ('%PREDNISON%') \
                   OR \
                   Product_Name LIKE ('%METHYLPRE%') \
                   group by Product_Name, Year, Quarter \
                   order by Year asc, Quarter asc, Product_Name asc, sum(Number_of_Prescriptions) desc"))

# COMMAND ----------

# 2020 only - Top Rx by drug - Albuterol + Corticosteroids

display(spark.sql("select Product_Name, Year, Quarter, sum(Number_of_Prescriptions) \
                   from SDUD2020_merge \
                   where Year = '2020' \
                   and (Product_Name = 'ALBUTEROL' \
                         OR \
                         Product_Name LIKE ('%DEXAMETHA%') \
                         OR \
                         Product_Name LIKE ('%PREDNISON%') \
                         OR \
                         Product_Name LIKE ('%METHYLPRED%')) \
                   group by Product_Name, Year, Quarter \
                   order by Year asc, Product_Name asc, sum(Number_of_Prescriptions) desc"))

# COMMAND ----------

# Top Rx by drug - Prednisone only (2018-2020)

display(spark.sql("select Product_Name, Year, Quarter, sum(Number_of_Prescriptions) \
                   from SDUD2020_merge \
                   where Product_Name LIKE ('%PREDNISON%') \
                   group by Product_Name, Year, Quarter \
                   order by Year asc, Quarter asc, Product_Name asc, sum(Number_of_Prescriptions) desc"))

# COMMAND ----------

# Most expensive Rx reimbursed by the govt (Medicaid)
# How much did state Medicaid programs pay for each drug?

display(spark.sql("select Product_Name, sum(Medicaid_Amount_Reimbursed) \
                   from SDUD2020_merge \
                   where Product_Name <> 'UNKNOWN' \
                   group by Product_Name \
                   order by sum(Medicaid_Amount_Reimbursed) desc"))

# COMMAND ----------

# Most expensive Rx reimbursed by the govt (Medicaid) - corticosteroids (2018-2020)
# How much did state Medicaid programs pay for each drug?

display(spark.sql("select Product_Name, Year, sum(Medicaid_Amount_Reimbursed) \
                   from SDUD2020_merge \
                   where Product_Name = 'ALBUTEROL' \
                   OR \
                   Product_Name LIKE ('%DEXAMETHA%') \
                   OR \
                   Product_Name LIKE ('%PREDNISON%') \
                   OR \
                   Product_Name LIKE ('%METHYLPRED%') \
                   group by Product_Name, Year \
                   order by Year asc, Product_Name asc, sum(Medicaid_Amount_Reimbursed) desc"))

# COMMAND ----------

# plot by state (2020 only)

display(spark.sql("select Product_Name, State, sum(Number_of_Prescriptions) \
                   from SDUD2020_merge \
                   where Year = '2020' \
                   and State <> 'XX' \
                   and (Product_Name = 'ALBUTEROL' \
                         OR \
                         Product_Name LIKE ('%DEXAMETHA%') \
                         OR \
                         Product_Name LIKE ('%PREDNISON%') \
                         OR \
                         Product_Name LIKE ('%METHYLPRED%')) \
                   group by Product_Name, State \
                   order by Product_Name asc, sum(Number_of_Prescriptions) desc"))

# COMMAND ----------

# convert to Pandas dataframe
df4 = df3.toPandas()

# COMMAND ----------

# explore the data - how many rows, columns?
df4.shape
