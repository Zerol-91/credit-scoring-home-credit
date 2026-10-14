-- Client-level features from monthly snapshots of previous credit cards.
-- Grain: one row per SK_ID_CURR that has at least one credit card monthly record.
-- Two-step aggregation: monthly snapshots -> card -> client.

CREATE OR REPLACE TABLE feat_cc AS
WITH
cc_train AS (
    -- Snapshots of our clients only (decision 13: drop test-set clients)
    SELECT *
    FROM credit_card_balance
    WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application_train)
),

monthly AS (
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        SK_DPD_DEF,
        -- Utilization = balance / limit. Negative balance (overpayment) counts as 0.
        -- A zero limit makes utilization undefined: NULL, skipped by AVG and MAX.
        -- Values above 1 are real (over-limit spending) and are kept.
        GREATEST(AMT_BALANCE, 0) / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0) AS utilization
    FROM cc_train
),

per_card AS (
    -- Step 1: monthly snapshots -> one row per card
    SELECT
        SK_ID_CURR,
        SK_ID_PREV,
        COUNT(*)          AS months_cnt,
        AVG(utilization)  AS avg_util,
        MAX(utilization)  AS max_util,
        MAX(SK_DPD_DEF)   AS max_dpd
    FROM monthly
    GROUP BY SK_ID_CURR, SK_ID_PREV
)

-- Step 2: cards -> one row per client
SELECT
    SK_ID_CURR,

    SUM(months_cnt)  AS cc_months_cnt,
    -- Average over cards, each card weighs equally regardless of its age
    AVG(avg_util)    AS cc_avg_utilization,
    MAX(max_util)    AS cc_max_utilization,
    MAX(max_dpd)     AS cc_max_dpd

FROM per_card
GROUP BY SK_ID_CURR;
