CREATE SCHEMA IF NOT EXISTS main;

DROP TABLE IF EXISTS raw_customers;

CREATE TABLE raw_customers (
    customer_id VARCHAR,
    gender VARCHAR,
    senior_citizen INTEGER,
    partner VARCHAR,
    dependents VARCHAR,
    tenure INTEGER,
    phone_service VARCHAR,
    multiple_lines VARCHAR,
    internet_service VARCHAR,
    online_security VARCHAR,
    online_backup VARCHAR,
    device_protection VARCHAR,
    tech_support VARCHAR,
    streaming_tv VARCHAR,
    streaming_movies VARCHAR,
    contract VARCHAR,
    paperless_billing VARCHAR,
    payment_method VARCHAR,
    monthly_charges DOUBLE,
    total_charges DOUBLE,
    churn INTEGER
);