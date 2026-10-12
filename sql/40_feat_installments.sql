-- Client-level payment discipline on previous Home Credit loans.
-- Grain: one row per SK_ID_CURR that has at least one scheduled installment.
-- Two-step aggregation: payment rows -> installment -> client.
-- One installment can be paid in several parts (several rows), so lateness and
-- underpayment are measured per installment, not per payment row.

CREATE OR REPLACE TABLE feat_installments AS
WITH
inst_train AS (
    -- Payments of our clients only (decision 13: drop test-set clients)
    SELECT *
    FROM installments_payments
    WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application_train)
),

per_installment AS (
    -- Step 1: payment rows -> one row per installment.
    -- Key: loan + schedule version + installment number. A new version means
    -- the schedule was changed, so each version is a separate obligation.
    -- AMT_INSTALMENT and DAYS_INSTALMENT are constant within the key.
    SELECT
        SK_ID_CURR,
        MAX(DAYS_INSTALMENT)                       AS days_due,
        MAX(AMT_INSTALMENT)                        AS amt_due,
        COALESCE(SUM(AMT_PAYMENT), 0)              AS amt_paid,
        -- The installment is settled by its last part; NULL if nothing was paid
        MAX(DAYS_ENTRY_PAYMENT) - MAX(DAYS_INSTALMENT) AS days_late_raw
    FROM inst_train
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER
),

flags AS (
    SELECT
        SK_ID_CURR,
        days_due,
        -- Early payment counts as 0 days late, so it cannot offset real lateness
        GREATEST(days_late_raw, 0) AS days_late,
        -- Unpaid installment counts as late
        CASE WHEN days_late_raw > 0 OR days_late_raw IS NULL THEN 1 ELSE 0 END AS is_late,
        -- Tolerance of 1 currency unit against floating-point noise
        CASE WHEN amt_paid < amt_due - 1 THEN 1 ELSE 0 END AS is_underpaid
    FROM per_installment
)

-- Step 2: installments -> one row per client
SELECT
    SK_ID_CURR,

    COUNT(*)            AS inst_cnt,
    AVG(is_late)        AS inst_share_late,
    AVG(days_late)      AS inst_avg_days_late,
    MAX(days_late)      AS inst_max_days_late,
    AVG(is_underpaid)   AS inst_share_underpaid,

    -- NULL when the client had no installments due in the last year
    AVG(is_late) FILTER (WHERE days_due >= -365) AS inst_share_late_last_year

FROM flags
GROUP BY SK_ID_CURR;
