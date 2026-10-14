-- Client-level features from monthly snapshots of previous POS and cash loans.
-- Grain: one row per SK_ID_CURR that has at least one POS/cash monthly record.
-- Two-step aggregation: monthly snapshots -> loan -> client.
-- DPD is taken from SK_DPD_DEF (DPD with tolerance: tiny debts are ignored).

CREATE OR REPLACE TABLE feat_pos AS
WITH
pos_train AS (
    -- Snapshots of our clients only (decision 13: drop test-set clients)
    SELECT *
    FROM pos_cash_balance
    WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application_train)
),

per_loan AS (
    -- Step 1: monthly snapshots -> one row per loan
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        COUNT(*)                                      AS months_cnt,
        COUNT(*) FILTER (WHERE SK_DPD_DEF > 0)        AS months_dpd,
        MAX(SK_DPD_DEF)                               AS max_dpd,
        -- 'Demand': the bank demanded early repayment of the whole debt
        MAX(CASE WHEN NAME_CONTRACT_STATUS = 'Demand' THEN 1 ELSE 0 END) AS ever_demand,
        -- Active at application date: the latest snapshot is the month before
        -- the application and says 'Active'. Many loans stop reporting earlier,
        -- so their last 'Active' status is outdated.
        CASE WHEN MAX(MONTHS_BALANCE) = -1
              AND arg_max(NAME_CONTRACT_STATUS, MONTHS_BALANCE) = 'Active'
             THEN 1 ELSE 0
        END                                           AS is_active
    FROM pos_train
    GROUP BY SK_ID_CURR, SK_ID_PREV
)

-- Step 2: loans -> one row per client
SELECT
    SK_ID_CURR,

    SUM(months_cnt)                   AS pos_months_cnt,
    MAX(max_dpd)                      AS pos_max_dpd,
    -- Pooled share: long loans weigh more than short ones (as in bureau)
    SUM(months_dpd) / SUM(months_cnt) AS pos_share_months_dpd,
    SUM(is_active)                    AS pos_active_cnt,
    MAX(ever_demand)                  AS pos_ever_demand

FROM per_loan
GROUP BY SK_ID_CURR;
