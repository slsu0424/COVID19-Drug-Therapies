# Databricks notebook source
# MAGIC %md #Part 1: Load data, Query, and Join

# COMMAND ----------

# MAGIC %md ##Set Spark Configuration

# COMMAND ----------

# Define access key to connect to adls2

# https://www.mssqltips.com/sqlservertip/6499/reading-and-writing-data-in-azure-data-lake-storage-gen-2-with-azure-databricks/

# http://peter.lalovsky.com/2021/07/azure/azure-databricks-read-write-files-from-to-azure-data-lake/

#spark.conf.set(
#  "fs.azure.account.key.sdudsynapseadls2.dfs.core.windows.net",
#  dbutils.secrets.get(scope="secret01",key="testsecret01")
#)

# COMMAND ----------

# MAGIC %md #Load data

# COMMAND ----------

# Load datasets from DBFS into Spark dataframe

# display(dbutils.fs.ls("/mnt/adls/FAERS_Output/ASCII"))

display(dbutils.fs.ls("/mnt/adls/FAERS_Output_v3/ASCII"))

# COMMAND ----------

# https://intellipaat.com/community/8588/how-to-import-multiple-csv-files-in-a-single-load

#df_demo = spark.read.csv("/mnt/adls/FAERS_Output/ASCII/DEMO*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # demographic
#df_drug = spark.read.csv("/mnt/adls/FAERS_Output/ASCII/DRUG*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # drug
#df_indi = spark.read.csv("/mnt/adls/FAERS_Output/ASCII/INDI*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # indication
#df_outc = spark.read.csv("/mnt/adls/FAERS_Output/ASCII/OUTC*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # outcome
#df_reac = spark.read.csv("/mnt/adls/FAERS_Output/ASCII/REAC*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # reaction
#df_rpsr = spark.read.csv("/mnt/adls/FAERS_Output/ASCII/RPSR*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # report sources
#df_ther = spark.read.csv("/mnt/adls/FAERS_Output/ASCII/THER*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # therapy

# display sample dataset
#display(df_demo)

# COMMAND ----------

# https://intellipaat.com/community/8588/how-to-import-multiple-csv-files-in-a-single-load

df_demo = spark.read.csv("/mnt/adls/FAERS_Output_v3/ASCII/DEMO*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # demographic
df_drug = spark.read.csv("/mnt/adls/FAERS_Output_v3/ASCII/DRUG*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # drug
df_indi = spark.read.csv("/mnt/adls/FAERS_Output_v3/ASCII/INDI*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # indication
df_outc = spark.read.csv("/mnt/adls/FAERS_Output_v3/ASCII/OUTC*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # outcome
df_reac = spark.read.csv("/mnt/adls/FAERS_Output_v3/ASCII/REAC*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # reaction
df_rpsr = spark.read.csv("/mnt/adls/FAERS_Output_v3/ASCII/RPSR*", header="true", nullValue = "NA", inferSchema="true", sep = "$") # report sources
df_ther = spark.read.csv("/mnt/adls/FAERS_Output_v3/ASCII/THER*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # therapy

# display sample dataset
display(df_demo)

# COMMAND ----------

# MAGIC %md ##Rows and Columns

# COMMAND ----------

# how many rows, columns in each dataset

print((df_demo.count(), len(df_demo.columns)))
print((df_drug.count(), len(df_drug.columns)))
print((df_indi.count(), len(df_indi.columns)))
print((df_outc.count(), len(df_outc.columns)))
print((df_reac.count(), len(df_reac.columns)))
print((df_rpsr.count(), len(df_rpsr.columns)))
print((df_ther.count(), len(df_ther.columns)))

# COMMAND ----------

# MAGIC %md ##Sample Schema

# COMMAND ----------

df_demo.printSchema()

# COMMAND ----------

# MAGIC %md #Query Datasets

# COMMAND ----------

# create temp views for all ASCII data files - individual tables

# https://analyticshut.com/running-sql-queries-on-spark-dataframes/
# https://towardsdatascience.com/pyspark-and-sparksql-basics-6cb4bf967e53
# https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.select.html

df_demo.createOrReplaceTempView("2020_Demo")
df_drug.createOrReplaceTempView("2020_Drug")
df_indi.createOrReplaceTempView("2020_Indi")
df_outc.createOrReplaceTempView("2020_Outc")
df_reac.createOrReplaceTempView("2020_Reac")
df_rpsr.createOrReplaceTempView("2020_Rpsr")
df_ther.createOrReplaceTempView("2020_Ther")

# COMMAND ----------

# MAGIC %md ##Demographic

# COMMAND ----------

#spark.sql("select * from 2020_Demo").show() 

#display(spark.sql("select * from 2020_Demo"))

display(spark.sql("select * from 2020_Demo where caseid = '18523926'"))

#display(spark.sql("select * from 2020_Demo where age is not null"))

#display(spark.sql("select occr_country, count(caseid) as Num_of_AE from 2020_Demo group by occr_country order by Num_of_AE desc"))

#use pyspark
#display(df_demo.select('*'))

# COMMAND ----------

# MAGIC %md ##Drug

# COMMAND ----------

display(spark.sql("select * from 2020_Drug"))

#display(spark.sql("select * from 2020_Drug where drugname = 'METHYLPREDNISOLONE.'"))

#display(spark.sql("select * from 2020_Drug where drugname IN ('DEXAMETHASONE.', 'PREDNISONE', 'PREDNISONE.', 'METHYLPREDNISOLONE.') order by drugname desc"))

#display(spark.sql("select * from 2020_Drug where prod_ai = 'ALBUTEROL'"))

#display(spark.sql("select * from 2020_Drug group where caseid = '17363177'"))

#display(spark.sql("select drugname, count(drugname) as Total from 2020_Drug group by drugname order by Total desc"))

# COMMAND ----------

# MAGIC %md ##Indication

# COMMAND ----------

display(spark.sql("select * from 2020_INDI"))

#display(spark.sql("select * from 2020_INDI where caseid = '17363177'"))

# COMMAND ----------

# MAGIC %md ##Outcome

# COMMAND ----------

#display(spark.sql("select * from 2020_OUTC"))

display(spark.sql("select outc_cod, count(outc_cod) from 2020_OUTC group by outc_cod order by count(outc_cod) desc"))

#display(spark.sql("select * from 2020_Outc where caseid = '17488958'"))

# COMMAND ----------

# drug & outcome

display(spark.sql("select A.drugname, A.prod_ai, B.outc_cod, count(B.outc_cod) \
                    from 2020_Drug A INNER JOIN 2020_Outc B \
                    on A.primaryid = B.primaryid \
                    where B.outc_cod = 'HO' \
                    group by A.drugname, A.prod_ai, B.outc_cod \
                    order by count(outc_cod) desc"))

# COMMAND ----------

# outcomes - Albuterol + Corticosteroids

display(spark.sql("select A.drugname, B.outc_cod, count(B.outc_cod) \
                    from 2020_Drug A INNER JOIN 2020_Outc B \
                    on A.primaryid = B.primaryid \
                    AND A.drugname IN ('ALBUTEROL.', 'DEXAMETHASONE.', 'PREDNISONE', 'METHYLPREDNISOLONE') \
                    group by A.drugname, B.outc_cod \
                    order by count(outc_cod) desc"))

# COMMAND ----------

# MAGIC %md ##Reaction

# COMMAND ----------

#display(spark.sql("select pt, count(pt) from 2020_REAC group by pt order by count(pt) desc"))

display(spark.sql("select * from 2020_REAC where caseid = '17356818'"))

# COMMAND ----------

# MAGIC %md ##Report Sources

# COMMAND ----------

display(spark.sql("select * from 2020_RPSR"))

# COMMAND ----------

# MAGIC %md ##Therapy

# COMMAND ----------

display(spark.sql("select * from 2020_Ther where caseid = '17363177' and dsg_drug_seq in ('11','12','13')"))

# COMMAND ----------

# join - for corticosteroids

# able to export all - remove dup columns

display(spark.sql("select D.*, \
                         DR.drug_seq, DR.role_cod, DR.drugname, DR.prod_ai, DR.val_vbm, DR.route, DR.dose_vbm, DR.cum_dose_chr, DR.cum_dose_unit, DR.dechal, DR.rechal, DR.lot_num, DR.exp_dt, DR.nda_num, DR.dose_amt, DR.dose_unit, DR.dose_form, DR.dose_freq, \
                         R.pt, R.drug_rec_act, \
                         O.outc_cod, \
                         T.dsg_drug_seq, T.start_dt, T.end_dt, T.dur, T.dur_cod \
                         from 2020_DEMO D INNER JOIN 2020_DRUG DR \
                         ON (D.primaryid = DR.primaryid AND D.caseid = DR.caseid) \
                         INNER JOIN 2020_REAC R \
                         ON (D.primaryid = R.primaryid AND D.caseid = R.caseid) \
                         INNER JOIN 2020_OUTC O \
                         ON (D.primaryid = O.primaryid AND D.caseid = O.caseid) \
                         INNER JOIN 2020_INDI I \
                         ON (DR.primaryid = I.primaryid AND DR.caseid = I.caseid) \
                         INNER JOIN 2020_THER T \
                         ON (DR.primaryid = T.primaryid AND DR.caseid = T.caseid and DR.drug_seq = T.dsg_drug_seq) \
                         WHERE D.age is not null \
                         AND \
                         D.event_dt > '20200101' \
                         AND \
                         O.outc_cod IN ('HO', 'DE') \
                         AND \
                         DR.drugname IN ('DEXAMETHASONE.', 'PREDNISONE', 'PREDNISONE.', 'METHYLPREDNISOLONE.')"))

# D.caseid = '17363177' \

# COMMAND ----------

display(spark.sql("select * from 2020_Demo where caseid = '17363177'"))

# COMMAND ----------

# MAGIC %md #Join Tables

# COMMAND ----------

# demo - get most recent caseversion (subquery)

display(spark.sql("select *, \
                   MAX(caseversion) OVER (PARTITION BY caseid) as last_case_version \
                   FROM 2020_DEMO \
                   WHERE caseid = '17363177' "))

# COMMAND ----------

# demo - select with subquery

display(spark.sql("SELECT * \
                    FROM \
                      (SELECT *, \
                      MAX(caseversion) OVER (PARTITION BY caseid) as last_case_version \
                      FROM 2020_DEMO \
                      WHERE caseid = '17363177') \
                    WHERE caseversion = last_case_version"))

# COMMAND ----------

# store as dataframe - single case

df_demo1 = spark.sql("SELECT * \
                      FROM \
                        (SELECT *, \
                        MAX(caseversion) OVER (PARTITION BY caseid) as last_case_version \
                        FROM 2020_DEMO \
                        WHERE caseid = '17363177') \
                      WHERE caseversion = last_case_version")

# COMMAND ----------

# store as dataframe - all

df_demo1 = spark.sql("SELECT * \
                      FROM \
                        (SELECT *, \
                        MAX(caseversion) OVER (PARTITION BY caseid) as last_case_version \
                        FROM 2020_DEMO) \
                      WHERE caseversion = last_case_version")

# COMMAND ----------

# create temp view

df_demo1.createOrReplaceTempView("2020_DEMO1")

# COMMAND ----------

# MAGIC %md ##Corticosteroids

# COMMAND ----------

# query #4 - exclude indications

df1_old = spark.sql("select D.*, \
                         DR.drug_seq, DR.role_cod, DR.drugname, DR.prod_ai, DR.val_vbm, DR.route, DR.dose_vbm, DR.cum_dose_chr, DR.cum_dose_unit, DR.dechal, DR.rechal, DR.lot_num, DR.exp_dt, DR.nda_num, DR.dose_amt, DR.dose_unit, DR.dose_form, DR.dose_freq, \
                         R.pt, R.drug_rec_act, \
                         O.outc_cod, \
                         T.dsg_drug_seq, T.start_dt, T.end_dt, T.dur, T.dur_cod \
                         from 2020_DEMO1 D INNER JOIN 2020_DRUG DR \
                         ON D.primaryid = DR.primaryid \
                         INNER JOIN 2020_REAC R \
                         ON D.primaryid = R.primaryid \
                         INNER JOIN 2020_OUTC O \
                         ON D.primaryid = O.primaryid \
                         INNER JOIN 2020_THER T \
                         ON (DR.primaryid = T.primaryid and DR.drug_seq = T.dsg_drug_seq) \
                         WHERE D.age is not null \
                         AND \
                         D.event_dt > '20200101' \
                         AND \
                         O.outc_cod IN ('HO', 'DE') \
                         AND \
                         DR.drugname IN ('DEXAMETHASONE.', 'PREDNISONE', 'PREDNISONE.', 'METHYLPREDNISOLONE.') \
                         AND DR.route NOT IN ('null' 'unknown') \
                         AND DR.dose_amt NOT IN ('null') ")

                         # AND DR.dose_vbm NOT IN ('null', 'UNK') ")
                         # AND DR.dose_amt NOT IN ('null') ")
                         # AND \
                         # D.caseid = '17363177' \

display(df1_old)

# COMMAND ----------

# final query #4
# exclude indications

df1 = spark.sql("select D.*, \
                         DR.drug_seq, DR.role_cod, DR.drugname, DR.prod_ai, DR.val_vbm, DR.route, DR.dose_vbm, DR.cum_dose_chr, DR.cum_dose_unit, DR.dechal, DR.rechal, DR.lot_num, DR.exp_dt, DR.nda_num, DR.dose_amt, DR.dose_unit, DR.dose_form, DR.dose_freq, \
                         R.pt, R.drug_rec_act, \
                         O.outc_cod, \
                         T.dsg_drug_seq, T.start_dt, T.end_dt, T.dur, T.dur_cod \
                         from 2020_DEMO1 D INNER JOIN 2020_DRUG DR \
                         ON D.primaryid = DR.primaryid \
                         INNER JOIN 2020_REAC R \
                         ON D.primaryid = R.primaryid \
                         INNER JOIN 2020_OUTC O \
                         ON D.primaryid = O.primaryid \
                         INNER JOIN 2020_THER T \
                         ON (DR.primaryid = T.primaryid and DR.drug_seq = T.dsg_drug_seq) \
                         WHERE \
                         DR.drugname IN ('DEXAMETHASONE.', 'PREDNISONE', 'PREDNISONE.', 'METHYLPREDNISOLONE.') ")

display(df1)

# COMMAND ----------

# how many rows, columns

print((df1.count(), len(df1.columns)))

# COMMAND ----------

# MAGIC %md ##Filter Event Dates

# COMMAND ----------

# https://www.geeksforgeeks.org/filtering-rows-based-on-column-values-in-pyspark-dataframe/

#df2 = df1.where((df1.event_dt>='20200101'))
df2 = df1.where((df1.event_dt>='20190101'))

display(df2)

# if using pandas
#df2 = df1[df1['event_dt'] >= 20200101]

#display(df2)

# COMMAND ----------

# how many rows, columns

print((df2.count(), len(df2.columns)))

# COMMAND ----------

# MAGIC %md #Save to ADLS

# COMMAND ----------

# convert to pandas

df3 = df2.toPandas()

#df2 = df1.toPandas()

# COMMAND ----------

df3.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess1.csv', index=False)

#df2.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess1.csv', index=False)

# COMMAND ----------

#df3.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess1_minwhereclauses.csv', index=False)
