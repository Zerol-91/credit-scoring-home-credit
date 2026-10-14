-- Client-level data mart: application features joined with all history sources.
-- Grain: one row per SK_ID_CURR from application_train (307 511 rows).
-- Every feature table is already at client grain, so LEFT JOINs cannot multiply rows.
-- A client without history in a source gets NULL features and a 0 history flag.

CREATE OR REPLACE TABLE data_mart AS
WITH
split AS (
    -- Stratified 60/20/20 split (decision: train / valid / test).
    -- Within each TARGET class, clients are ordered by a hash of their id,
    -- so the split is random-like, reproducible and keeps the bad rate equal.
    -- md5 is used instead of hash(): its output does not change between DuckDB versions.
    SELECT
        SK_ID_CURR,
        CASE
            WHEN rn <= 0.6 * n THEN 'train'
            WHEN rn <= 0.8 * n THEN 'valid'
            ELSE 'test'
        END AS sample
    FROM (
        SELECT
            SK_ID_CURR,
            ROW_NUMBER() OVER (PARTITION BY TARGET ORDER BY md5(CAST(SK_ID_CURR AS VARCHAR))) AS rn,
            COUNT(*)     OVER (PARTITION BY TARGET) AS n
        FROM application_train
    )
)

SELECT
    s.sample,
    a.*,

    -- History flags: absence of data in a source is information in itself
    CASE WHEN b.SK_ID_CURR  IS NULL THEN 0 ELSE 1 END AS has_bureau_history,
    CASE WHEN p.SK_ID_CURR  IS NULL THEN 0 ELSE 1 END AS has_prev_history,
    CASE WHEN i.SK_ID_CURR  IS NULL THEN 0 ELSE 1 END AS has_inst_history,
    CASE WHEN ps.SK_ID_CURR IS NULL THEN 0 ELSE 1 END AS has_pos_history,
    CASE WHEN c.SK_ID_CURR  IS NULL THEN 0 ELSE 1 END AS has_cc_history,

    b.* EXCLUDE (SK_ID_CURR),
    b.bureau_active_debt / NULLIF(a.AMT_INCOME_TOTAL, 0) AS bureau_debt_to_income,
    p.* EXCLUDE (SK_ID_CURR),
    i.* EXCLUDE (SK_ID_CURR),
    ps.* EXCLUDE (SK_ID_CURR),
    c.* EXCLUDE (SK_ID_CURR)

FROM feat_application        AS a
JOIN split                   AS s  ON s.SK_ID_CURR  = a.SK_ID_CURR
LEFT JOIN feat_bureau        AS b  ON b.SK_ID_CURR  = a.SK_ID_CURR
LEFT JOIN feat_prev          AS p  ON p.SK_ID_CURR  = a.SK_ID_CURR
LEFT JOIN feat_installments  AS i  ON i.SK_ID_CURR  = a.SK_ID_CURR
LEFT JOIN feat_pos           AS ps ON ps.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN feat_cc            AS c  ON c.SK_ID_CURR  = a.SK_ID_CURR;
