if not exists(select * from sys.database_principals where name = 'asa.sql.workload01') create user [asa.sql.workload01] from login [asa.sql.workload01]
if not exists(select * from sys.database_principals where name = 'asa.sql.workload02') create user [asa.sql.workload02] from login [asa.sql.workload02]
if not exists(select * from sys.database_principals where name = 'ceo') create user [CEO] without login;
execute sp_addrolemember 'db_datareader', 'asa.sql.workload01' 
execute sp_addrolemember 'db_datareader', 'asa.sql.workload02' 
execute sp_addrolemember 'db_datareader', 'CEO' 
if not exists(select * from sys.database_principals where name = 'DataAnalystMiami') create user [DataAnalystMiami] without login
if not exists(select * from sys.database_principals where name = 'DataAnalystSanDiego')create user [DataAnalystSanDiego] without login
if not exists(select * from sys.schemas where name='wwi_mcw') EXEC('create schema [wwi_mcw] authorization [dbo]')
create master key
create table [wwi_mcw].[Product](ProductId SMALLINT NOT NULL,Seasonality TINYINT NOT NULL,Price DECIMAL(6,2),Profit DECIMAL(6,2))WITH(DISTRIBUTION = REPLICATE)
CREATE TABLE [wwi_mcw].[ProductQuantityForecast]([ProductId] [int]  NOT NULL, [TransactionDate] [int]  NOT NULL, [Hour] [int]  NOT NULL,[TotalQuantity] [int]  NOT NULL) WITH ( DISTRIBUTION = REPLICATE)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (100, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (200, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (300, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (400, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (500, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (600, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (700, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (800, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (900, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (1000, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (1100, 20201209, 10, 0)
INSERT INTO [wwi_mcw].[ProductQuantityForecast] VALUES (1200, 20201209, 10, 0)