CREATE OR REPLACE VIEW customer_features AS
SELECT
    customer_id,
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
    churn,

    CASE
        WHEN tenure < 12 THEN 'new'
        WHEN tenure < 24 THEN 'mid_term'
        ELSE 'long_term'
    END AS tenure_group,

    CASE
        WHEN contract IN ('One year', 'Two year') THEN 1
        ELSE 0
    END AS long_contract,

    CASE
        WHEN monthly_charges >= 70 THEN 1
        ELSE 0
    END AS high_monthly_charges,

    CASE
        WHEN tenure > 0 THEN total_charges / tenure
        ELSE NULL
    END AS avg_monthly_value_proxy,

    (
        CASE WHEN online_security = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN online_backup = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN tech_support = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN streaming_tv = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END
    ) AS subscribed_services_count

FROM raw_customers;