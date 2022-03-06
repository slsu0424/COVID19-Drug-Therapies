IF NOT EXISTS (SELECT * FROM sys.objects O JOIN sys.schemas S ON O.schema_id = S.schema_id WHERE O.NAME = 'SDUD2019' AND O.TYPE = 'U' AND S.NAME = 'dbo')
CREATE TABLE dbo.SDUD2019
	(
	 [Utilization_Type] nvarchar(4000),
	 [State] nvarchar(4000),
	 [Labeler_Code] bigint,
	 [Product_Code] bigint,
	 [Package_Size] bigint,
	 [Year] bigint,
	 [Quarter] bigint,
	 [Product_Name] nvarchar(4000),
	 [Suppression_Used] bit,
	 [Units_Reimbursed] nvarchar(4000),
	 [Number_of_Prescriptions] nvarchar(4000),
	 [Total_Amount_Reimbursed] nvarchar(4000),
	 [Medicaid_Amount_Reimbursed] nvarchar(4000),
	 [Non_Medicaid_Amount_Reimbursed] nvarchar(4000),
	 [Quarter_Begin] nvarchar(4000),
	 [Quarter_Begin_Date] nvarchar(4000),
	 [Latitude] float,
	 [Longitude] float,
	 [Location] nvarchar(4000),
	 [NDC] bigint
	)
WITH
	(
	DISTRIBUTION = ROUND_ROBIN,
	 HEAP
	 -- CLUSTERED COLUMNSTORE INDEX
	)
GO

--Uncomment the 4 lines below to create a stored procedure for data pipeline orchestration​
--CREATE PROC bulk_load_SDUD2019
--AS
--BEGIN
COPY INTO dbo.SDUD2018
(Utilization_Type 1, State 2, Labeler_Code 3, Product_Code 4, Package_Size 5, Year 6, Quarter 7, Product_Name 8, Suppression_Used 9, Units_Reimbursed 10, Number_of_Prescriptions 11, Total_Amount_Reimbursed 12, Medicaid_Amount_Reimbursed 13, Non_Medicaid_Amount_Reimbursed 14, Quarter_Begin 15, Quarter_Begin_Date 16, Latitude 17, Longitude 18, Location 19, NDC 20)
FROM 'https://sdudsynapseadls2.dfs.core.windows.net/sdudsynapsefilesystem/State_Drug_Utilization_Data_2019.csv'
WITH
(
	FILE_TYPE = 'CSV'
	,MAXERRORS = 0
	,FIRSTROW = 2
	,ERRORFILE = 'https://sdudsynapseadls2.dfs.core.windows.net/sdudsynapsefilesystem/'
)
--END
GO

SELECT TOP 100 * FROM dbo.SDUD2019
GO