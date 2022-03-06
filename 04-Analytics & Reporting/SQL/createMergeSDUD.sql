SELECT *
  INTO  mergeSDUD 
FROM
(
    SELECT *
        FROM SDUD2018
    UNION ALL
        SELECT *
            FROM SDUD2019
    UNION ALL
        SELECT *
            FROM SDUD2020
) a