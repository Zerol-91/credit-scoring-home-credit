-- Client-level features from the credit bureau.
-- Grain: one row per SK_ID_CURR that has at least one bureau record.
-- Two-step aggregation: bureau_balance (months) -> credit -> client.

CREATE OR REPLACE TABLE feat_bureau AS
WITH
train_ids AS (
    SELECT SK_ID_CURR
    FROM application_train
),

bureau_train AS (
    -- Bureau credits of our clients only (decision 2A: drop test-set clients)
    SELECT *
    FROM bureau
    WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM train_ids)
),

balance_per_credit AS (
    -- Step 1: monthly history -> one row per bureau credit
    -- STATUS: '0' no DPD, '1'..'5' DPD buckets, 'C' closed, 'X' unknown
    SELECT
        SK_ID_BUREAU,
        MAX(CASE WHEN STATUS IN ('1', '2', '3', '4', '5')
                 THEN CAST(STATUS AS INTEGER)
                 ELSE 0
            END) AS worst_status,
        COUNT(*) FILTER (WHERE STATUS IN ('0', '1', '2', '3', '4', '5')) AS months_known,
        COUNT(*) FILTER (WHERE STATUS IN ('1', '2', '3', '4', '5'))      AS months_dpd
    FROM bureau_balance
    WHERE SK_ID_BUREAU IN (SELECT SK_ID_BUREAU FROM bureau_train)
    GROUP BY SK_ID_BUREAU
)

-- Step 2: credits -> one row per client
SELECT
    b.SK_ID_CURR,

    COUNT(*)                                            AS bureau_cnt,
    COUNT(*) FILTER (WHERE b.CREDIT_ACTIVE = 'Active')  AS bureau_active_cnt,
    COALESCE(SUM(b.AMT_CREDIT_SUM_DEBT)
             FILTER (WHERE b.CREDIT_ACTIVE = 'Active'), 0) AS bureau_active_debt,

    MAX(CASE WHEN b.CREDIT_DAY_OVERDUE > 0 OR b.AMT_CREDIT_SUM_OVERDUE > 0
             THEN 1 ELSE 0
        END)                                            AS bureau_overdue_now,

    -MAX(b.DAYS_CREDIT)                                 AS bureau_days_since_last_credit,
    COUNT(*) FILTER (WHERE b.DAYS_CREDIT >= -365)       AS bureau_credits_last_year,

    MAX(p.worst_status)                                 AS bureau_worst_status,
    SUM(p.months_dpd) / NULLIF(SUM(p.months_known), 0)  AS bureau_share_months_dpd

FROM bureau_train AS b
LEFT JOIN balance_per_credit AS p
       ON p.SK_ID_BUREAU = b.SK_ID_BUREAU
GROUP BY b.SK_ID_CURR;