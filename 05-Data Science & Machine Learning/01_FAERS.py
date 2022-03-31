# Databricks notebook source
# MAGIC %md
# MAGIC #Part 1: Curate Corticosteroids Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC # Update Variables

# COMMAND ----------

service_principal_client_id = '34e3f258-4f4f-404a-8c60-8cc2eab1ac60' # app registration app (client) id
tenant_id = '72f988bf-86f1-41af-91ab-2d7cd011db47' # app registration tenant id
storage_account_name = 'asastgssuaefdbhdg2dbc4'
container_name = 'curated'

# COMMAND ----------

# MAGIC %md
# MAGIC # Use Azure Key Vault and Databricks Scope for Secure Access
# MAGIC 
# MAGIC By interacting with Azure Key Vault, you can access your storage without writing secure information in Databricks.
# MAGIC 
# MAGIC Before starting:
# MAGIC 1. Create your key vault resource in Azure Portal
# MAGIC 2. Create new secret and set service principal's secret as value in your key vault. Here we assume the key name is "testsecret01".
# MAGIC 2. Go to https://YOUR_AZURE_DATABRICKS_URL#secrets/createScope    
# MAGIC (Once you've created scope, you should manage with Databricks CLI.)    
# MAGIC     - Set scope name. Here we assume the name is "scope01", which is needed for the following "dbutils" commands.
# MAGIC     - Select "Creator" (needing Azure Databricks Premium tier)
# MAGIC     - Input your key vault's settings, such as DNS name and resource id. (You can copy settings in key vault's "Properties".)

# COMMAND ----------

# get mount points

for mount in dbutils.fs.mounts():
    print(mount.mountPoint)

# COMMAND ----------

# https://transform365.blog/2020/06/15/mount-a-blob-storage-in-azure-databricks-only-if-it-does-not-exist-using-python/
# https://docs.microsoft.com/en-us/azure/databricks/security/secrets/secret-scopes

sp_secret = dbutils.secrets.get(scope = "scope01", key = "testsecret01")

configs = {"fs.azure.account.auth.type": "OAuth",
           "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
           "fs.azure.account.oauth2.client.id": service_principal_client_id,
           "fs.azure.account.oauth2.client.secret": sp_secret,
           "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/" + tenant_id + "/oauth2/token"}

try:
    dbutils.fs.mount(
    source = "abfss://{}@{}.dfs.core.windows.net".format(container_name, storage_account_name),
    mount_point = "/mnt/adls",
    extra_configs = configs
    )
except Exception as e:
    print("already mounted. Try to unmount first")

# COMMAND ----------

# unmount folders as needed

dbutils.fs.unmount("/mnt/adls")

# COMMAND ----------

# MAGIC %md
# MAGIC #Load data

# COMMAND ----------

# view folders and files from mounted ADLS container in DBFS

display(dbutils.fs.ls("/mnt/adls/FAERS_output"))

# COMMAND ----------

# view files in the ASCII folder

display(dbutils.fs.ls("/mnt/adls/FAERS_output/ASCII"))

# COMMAND ----------

# Load datasets into Spark dataframe

# https://intellipaat.com/community/8588/how-to-import-multiple-csv-files-in-a-single-load

df_demo = spark.read.csv("/mnt/adls/FAERS_output/ASCII/DEMO*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # demographic
df_drug = spark.read.csv("/mnt/adls/FAERS_output/ASCII/DRUG*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # drug
df_indi = spark.read.csv("/mnt/adls/FAERS_output/ASCII/INDI*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # indication
df_outc = spark.read.csv("/mnt/adls/FAERS_output/ASCII/OUTC*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # outcome
df_reac = spark.read.csv("/mnt/adls/FAERS_output/ASCII/REAC*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # reaction
df_rpsr = spark.read.csv("/mnt/adls/FAERS_output/ASCII/RPSR*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # report sources
df_ther = spark.read.csv("/mnt/adls/FAERS_output/ASCII/THER*.txt", header="true", nullValue = "NA", inferSchema="true", sep = "$") # therapy

# display sample dataset
display(df_demo)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Rows and Columns

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

# MAGIC %md
# MAGIC ##Sample Schema

# COMMAND ----------

df_demo.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #Query Datasets

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

# MAGIC %md
# MAGIC ##Demographic

# COMMAND ----------

#spark.sql("select * from 2020_Demo").show() 

#display(spark.sql("select * from 2020_Demo"))

display(spark.sql("select * from 2020_Demo where caseid = '18523926'"))

#display(spark.sql("select * from 2020_Demo where age is not null"))

#display(spark.sql("select occr_country, count(caseid) as Num_of_AE from 2020_Demo group by occr_country order by Num_of_AE desc"))

#use pyspark
#display(df_demo.select('*'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Drug

# COMMAND ----------

display(spark.sql("select * from 2020_Drug"))

#display(spark.sql("select * from 2020_Drug where drugname = 'METHYLPREDNISOLONE.'"))

#display(spark.sql("select * from 2020_Drug where drugname IN ('DEXAMETHASONE.', 'PREDNISONE', 'PREDNISONE.', 'METHYLPREDNISOLONE.') order by drugname desc"))

#display(spark.sql("select * from 2020_Drug where prod_ai = 'ALBUTEROL'"))

#display(spark.sql("select * from 2020_Drug group where caseid = '17363177'"))

#display(spark.sql("select drugname, count(drugname) as Total from 2020_Drug group by drugname order by Total desc"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Indication

# COMMAND ----------

display(spark.sql("select * from 2020_INDI"))

#display(spark.sql("select * from 2020_INDI where caseid = '17363177'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Outcome

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

# MAGIC %md
# MAGIC ##Reaction

# COMMAND ----------

#display(spark.sql("select pt, count(pt) from 2020_REAC group by pt order by count(pt) desc"))

display(spark.sql("select * from 2020_REAC where caseid = '17356818'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Report Sources

# COMMAND ----------

display(spark.sql("select * from 2020_RPSR"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Therapy

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

# MAGIC %md
# MAGIC #Join Tables

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

# MAGIC %md
# MAGIC ##Corticosteroids

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

# MAGIC %md
# MAGIC ##Filter Event Dates

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

# MAGIC %md
# MAGIC #Save to ADLS

# COMMAND ----------

# convert to pandas

df3 = df2.toPandas()

#df2 = df1.toPandas()

# COMMAND ----------

df3.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess1.csv', index=False)

#df2.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess1.csv', index=False)

# COMMAND ----------

#df3.to_csv('/dbfs/mnt/adls/FAERS_CSteroid_preprocess1_minwhereclauses.csv', index=False)
