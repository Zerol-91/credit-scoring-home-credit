-- Client-level features from previous Home Credit applications.
-- Grain: one row per SK_ID_CURR that has at least one previous application.
-- DAYS_DECISION has no 365243 placeholder, so no NULLIF is needed here.

CREATE OR REPLACE TABLE feat_prev AS
WITH
prev_train AS (
    -- Previous applications of our clients only (decision 13: drop test-set clients)
    SELECT *
    FROM previous_application
    WHERE SK_ID_CURR IN (SELECT SK_ID_CURR FROM application_train)
)

SELECT
    SK_ID_CURR,

    COUNT(*)                                                  AS prev_cnt,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_approved_cnt,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused')  AS prev_refused_cnt,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused')
        / COUNT(*)                                            AS prev_refused_share,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused'
                       AND DAYS_DECISION >= -365)             AS prev_refused_last_year,

    -MAX(DAYS_DECISION)                                       AS prev_days_since_last_decision,

    -- Granted / requested amount: below 1 means the bank cut the requested sum.
    -- Only approved applications have a meaningful granted amount;
    -- AMT_APPLICATION = 0 (mostly cancelled applications) is excluded by NULLIF.
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))
        FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved')      AS prev_credit_to_application

FROM prev_train
GROUP BY SK_ID_CURR;
