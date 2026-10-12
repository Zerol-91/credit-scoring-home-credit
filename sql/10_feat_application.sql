-- Client-level features from the application form.
-- Grain: one row per SK_ID_CURR (application_train is already at this grain).

CREATE OR REPLACE TABLE feat_application AS
SELECT
    SK_ID_CURR,
    TARGET,

    -- Loan parameters and affordability
    NAME_CONTRACT_TYPE,
    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    AMT_ANNUITY,
    AMT_CREDIT  / NULLIF(AMT_INCOME_TOTAL, 0) AS credit_to_income,
    AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0) AS annuity_to_income,
    AMT_CREDIT  / NULLIF(AMT_ANNUITY, 0)      AS credit_term,
    AMT_CREDIT  / NULLIF(AMT_GOODS_PRICE, 0)  AS credit_to_goods,

    -- Demographics and employment
    -DAYS_BIRTH / 365.25 AS age_years,
    CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL
         ELSE -DAYS_EMPLOYED / 365.25
    END AS employment_years,
    CASE WHEN DAYS_EMPLOYED = 365243 THEN 1 ELSE 0 END AS days_employed_anomaly,
    NAME_EDUCATION_TYPE,
    NAME_INCOME_TYPE,
    NAME_FAMILY_STATUS,
    CNT_CHILDREN,
    FLAG_OWN_CAR,
    FLAG_OWN_REALTY,
    OCCUPATION_TYPE,
    ORGANIZATION_TYPE,
    REGION_RATING_CLIENT,

    -- Stability signals
    -DAYS_ID_PUBLISH        / 365.25 AS years_since_id_change,
    -DAYS_LAST_PHONE_CHANGE / 365.25 AS years_since_phone_change,

    -- External scores and credit-seeking behaviour
    EXT_SOURCE_1,
    EXT_SOURCE_2,
    EXT_SOURCE_3,
    AMT_REQ_CREDIT_BUREAU_YEAR,

    -- Not a model feature: kept only for the fairness audit
    CODE_GENDER
FROM application_train;