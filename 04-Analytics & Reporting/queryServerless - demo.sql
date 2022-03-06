-- select top (csv)
SELECT 
    top 10 *
FROM OPENROWSET(
        BULK 'https://sdudsynapseadls2.dfs.core.windows.net/sdudsynapsefilesystem/State_Drug_Utilization_Data_2019.csv',
        FORMAT = 'CSV', PARSER_VERSION = '2.0',
        HEADER_ROW=TRUE
    ) as r