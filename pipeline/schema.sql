-- PostgreSQL DDL Schema for Churn Compass Synthetic Banking Data Simulator

DROP TABLE IF EXISTS churn_feature_snapshot CASCADE;
DROP TABLE IF EXISTS customer_churn_label CASCADE;
DROP TABLE IF EXISTS churn_simulation_state CASCADE;
DROP TABLE IF EXISTS customer_feedback CASCADE;
DROP TABLE IF EXISTS customer_complaints CASCADE;
DROP TABLE IF EXISTS digital_engagement_monthly CASCADE;
DROP TABLE IF EXISTS customer_monthly_activity CASCADE;
DROP TABLE IF EXISTS transaction_fact CASCADE;
DROP TABLE IF EXISTS product_holdings_monthly CASCADE;
DROP TABLE IF EXISTS loan_monthly_snapshot CASCADE;
DROP TABLE IF EXISTS loan_master CASCADE;
DROP TABLE IF EXISTS card_monthly_snapshot CASCADE;
DROP TABLE IF EXISTS card_portfolio CASCADE;
DROP TABLE IF EXISTS account_monthly_snapshot CASCADE;
DROP TABLE IF EXISTS account_master CASCADE;
DROP TABLE IF EXISTS customer_master CASCADE;
DROP TABLE IF EXISTS branch_master CASCADE;

-- 1. branch_master
CREATE TABLE branch_master (
    branch_code VARCHAR(20) PRIMARY KEY,
    branch_name VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    region VARCHAR(50) NOT NULL,
    branch_type VARCHAR(30) NOT NULL,
    open_date DATE NOT NULL,
    closure_date DATE NULL,
    customer_weight INT NOT NULL
);

-- 2. customer_master
CREATE TABLE customer_master (
    customer_id BIGINT PRIMARY KEY,
    cif_number VARCHAR(20) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(20) NOT NULL,
    marital_status VARCHAR(20) NOT NULL,
    occupation VARCHAR(100) NOT NULL,
    employment_type VARCHAR(50) NOT NULL,
    annual_income NUMERIC(18,2) NOT NULL,
    customer_since DATE NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    country VARCHAR(100) NOT NULL,
    kyc_status VARCHAR(20) NOT NULL,
    is_active BOOLEAN NOT NULL
);

-- 3. account_master
CREATE TABLE account_master (
    account_id BIGINT PRIMARY KEY,
    customer_id BIGINT REFERENCES customer_master(customer_id),
    branch_code VARCHAR(20) REFERENCES branch_master(branch_code),
    account_type VARCHAR(50) NOT NULL,
    open_date DATE NOT NULL,
    account_status VARCHAR(20) NOT NULL,
    account_currency VARCHAR(3) NOT NULL,
    salary_account_flag BOOLEAN NOT NULL,
    overdraft_limit NUMERIC(18,2) NOT NULL,
    account_close_date DATE NULL
);

-- 4. account_monthly_snapshot
CREATE TABLE account_monthly_snapshot (
    account_id BIGINT REFERENCES account_master(account_id),
    snapshot_month DATE NOT NULL,
    current_balance NUMERIC(18,2) NOT NULL,
    average_monthly_balance NUMERIC(18,2) NOT NULL,
    min_balance_30d NUMERIC(18,2) NOT NULL,
    max_balance_30d NUMERIC(18,2) NOT NULL,
    deposit_count INT NOT NULL,
    withdrawal_count INT NOT NULL,
    debit_txn_count INT NOT NULL,
    credit_txn_count INT NOT NULL,
    fee_charged_amount NUMERIC(18,2) NOT NULL,
    salary_credit_amount NUMERIC(18,2) NOT NULL,
    account_status VARCHAR(20) NOT NULL,
    PRIMARY KEY (account_id, snapshot_month)
);

-- 5. card_portfolio
CREATE TABLE card_portfolio (
    card_id BIGINT PRIMARY KEY,
    customer_id BIGINT REFERENCES customer_master(customer_id),
    card_type VARCHAR(20) NOT NULL,
    network VARCHAR(20) NOT NULL,
    issue_date DATE NOT NULL,
    expiry_date DATE NOT NULL,
    card_status VARCHAR(20) NOT NULL,
    primary_card_flag BOOLEAN NOT NULL,
    credit_limit NUMERIC(18,2) NOT NULL,
    rewards_program VARCHAR(50) NOT NULL,
    reward_tier VARCHAR(20) NOT NULL
);

-- 6. card_monthly_snapshot
CREATE TABLE card_monthly_snapshot (
    card_id BIGINT REFERENCES card_portfolio(card_id),
    snapshot_month DATE NOT NULL,
    monthly_spend_amount NUMERIC(18,2) NOT NULL,
    monthly_txn_count INT NOT NULL,
    cash_advance_amount NUMERIC(18,2) NOT NULL,
    utilization_rate NUMERIC(6,4) NOT NULL,
    min_due_amount NUMERIC(18,2) NOT NULL,
    payment_made_amount NUMERIC(18,2) NOT NULL,
    rewards_points_balance BIGINT NOT NULL,
    card_status VARCHAR(20) NOT NULL,
    delinquency_flag BOOLEAN NOT NULL,
    PRIMARY KEY (card_id, snapshot_month)
);

-- 7. loan_master
CREATE TABLE loan_master (
    loan_id BIGINT PRIMARY KEY,
    customer_id BIGINT REFERENCES customer_master(customer_id),
    branch_code VARCHAR(20) REFERENCES branch_master(branch_code),
    loan_type VARCHAR(50) NOT NULL,
    sanctioned_amount NUMERIC(18,2) NOT NULL,
    disbursement_date DATE NOT NULL,
    interest_rate NUMERIC(6,3) NOT NULL,
    tenure_months INT NOT NULL,
    emi_amount NUMERIC(18,2) NOT NULL,
    loan_purpose VARCHAR(100) NOT NULL,
    origination_channel VARCHAR(50) NOT NULL,
    loan_status VARCHAR(20) NOT NULL,
    maturity_date DATE NOT NULL
);

-- 8. loan_monthly_snapshot
CREATE TABLE loan_monthly_snapshot (
    loan_id BIGINT REFERENCES loan_master(loan_id),
    snapshot_month DATE NOT NULL,
    outstanding_balance NUMERIC(18,2) NOT NULL,
    emi_amount NUMERIC(18,2) NOT NULL,
    dpd_days INT NOT NULL,
    overdue_amount NUMERIC(18,2) NOT NULL,
    principal_paid_amount NUMERIC(18,2) NOT NULL,
    interest_paid_amount NUMERIC(18,2) NOT NULL,
    installment_due_amount NUMERIC(18,2) NOT NULL,
    installment_paid_amount NUMERIC(18,2) NOT NULL,
    loan_status VARCHAR(20) NOT NULL,
    restructuring_flag BOOLEAN NOT NULL,
    PRIMARY KEY (loan_id, snapshot_month)
);

-- 9. product_holdings_monthly
CREATE TABLE product_holdings_monthly (
    customer_id BIGINT REFERENCES customer_master(customer_id),
    snapshot_month DATE NOT NULL,
    savings_account_flag BOOLEAN NOT NULL,
    current_account_flag BOOLEAN NOT NULL,
    credit_card_flag BOOLEAN NOT NULL,
    personal_loan_flag BOOLEAN NOT NULL,
    home_loan_flag BOOLEAN NOT NULL,
    fixed_deposit_flag BOOLEAN NOT NULL,
    insurance_flag BOOLEAN NOT NULL,
    mutual_fund_flag BOOLEAN NOT NULL,
    demat_account_flag BOOLEAN NOT NULL,
    wealth_management_flag BOOLEAN NOT NULL,
    products_count INT NOT NULL,
    PRIMARY KEY (customer_id, snapshot_month)
);

-- 10. transaction_fact
CREATE TABLE transaction_fact (
    transaction_id BIGINT PRIMARY KEY,
    account_id BIGINT REFERENCES account_master(account_id),
    customer_id BIGINT REFERENCES customer_master(customer_id),
    txn_timestamp TIMESTAMP NOT NULL,
    txn_date DATE NOT NULL,
    txn_month DATE NOT NULL,
    txn_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    channel VARCHAR(30) NOT NULL,
    amount NUMERIC(18,2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    merchant_category VARCHAR(100) NOT NULL,
    merchant_name VARCHAR(150) NOT NULL,
    counterparty_type VARCHAR(50) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    is_salary_credit BOOLEAN NOT NULL,
    is_fee BOOLEAN NOT NULL,
    is_reversal BOOLEAN NOT NULL,
    balance_after_txn NUMERIC(18,2) NOT NULL
);

-- 11. customer_monthly_activity
CREATE TABLE customer_monthly_activity (
    customer_id BIGINT REFERENCES customer_master(customer_id),
    snapshot_month DATE NOT NULL,
    login_count INT NOT NULL,
    mobile_app_sessions INT NOT NULL,
    internet_banking_sessions INT NOT NULL,
    atm_transactions INT NOT NULL,
    branch_visits INT NOT NULL,
    debit_txn_count INT NOT NULL,
    credit_txn_count INT NOT NULL,
    total_debit_amount NUMERIC(18,2) NOT NULL,
    total_credit_amount NUMERIC(18,2) NOT NULL,
    avg_transaction_value NUMERIC(18,2) NOT NULL,
    unique_merchants INT NOT NULL,
    cash_withdrawal_count INT NOT NULL,
    card_present_txn_count INT NOT NULL,
    card_not_present_txn_count INT NOT NULL,
    days_since_last_txn INT NOT NULL,
    days_since_last_login INT NOT NULL,
    PRIMARY KEY (customer_id, snapshot_month)
);

-- 12. digital_engagement_monthly
CREATE TABLE digital_engagement_monthly (
    customer_id BIGINT REFERENCES customer_master(customer_id),
    snapshot_month DATE NOT NULL,
    mobile_app_active BOOLEAN NOT NULL,
    last_login_date DATE NOT NULL,
    push_notifications_sent INT NOT NULL,
    push_notifications_opened INT NOT NULL,
    email_campaigns_sent INT NOT NULL,
    email_clicks INT NOT NULL,
    campaigns_received INT NOT NULL,
    campaigns_responded INT NOT NULL,
    web_sessions INT NOT NULL,
    notification_opt_in BOOLEAN NOT NULL,
    app_crash_count INT NOT NULL,
    PRIMARY KEY (customer_id, snapshot_month)
);

-- 13. customer_complaints
CREATE TABLE customer_complaints (
    complaint_id BIGINT PRIMARY KEY,
    customer_id BIGINT REFERENCES customer_master(customer_id),
    complaint_date DATE NOT NULL,
    complaint_month DATE NOT NULL,
    channel VARCHAR(50) NOT NULL,
    category VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    resolution_days INT NULL,
    resolved_flag BOOLEAN NOT NULL,
    escalated_flag BOOLEAN NOT NULL,
    csat_score INT NULL,
    root_cause VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL
);

-- 14. customer_feedback
CREATE TABLE customer_feedback (
    feedback_id BIGINT PRIMARY KEY,
    customer_id BIGINT REFERENCES customer_master(customer_id),
    feedback_date DATE NOT NULL,
    feedback_month DATE NOT NULL,
    survey_channel VARCHAR(50) NOT NULL,
    survey_topic VARCHAR(100) NOT NULL,
    nps_score INT NULL,
    csat_score INT NULL
);

-- 15. churn_simulation_state
CREATE TABLE churn_simulation_state (
    customer_id BIGINT PRIMARY KEY REFERENCES customer_master(customer_id),
    persona VARCHAR(50) NOT NULL,
    low_sensitivity_segment BOOLEAN NOT NULL,
    churn_month DATE NULL,
    churned_flag BOOLEAN NOT NULL,
    churn_reason VARCHAR(100) NULL,
    active_months_generated INT NOT NULL
);

-- 16. customer_churn_label
CREATE TABLE customer_churn_label (
    customer_id BIGINT REFERENCES customer_master(customer_id),
    as_of_month DATE NOT NULL,
    prediction_horizon_months INT NOT NULL,
    churned BOOLEAN NOT NULL,
    churn_date DATE NULL,
    churn_reason VARCHAR(100) NULL,
    PRIMARY KEY (customer_id, as_of_month, prediction_horizon_months)
);

-- 17. churn_feature_snapshot
CREATE TABLE churn_feature_snapshot (
    customer_id BIGINT REFERENCES customer_master(customer_id),
    as_of_month DATE NOT NULL,
    prediction_horizon_months INT NOT NULL,
    tenure_months INT NOT NULL,
    products_count INT NOT NULL,
    balance_change_3m NUMERIC(18,4) NULL,
    txn_count_change_3m NUMERIC(18,4) NULL,
    login_count_change_6m NUMERIC(18,4) NULL,
    complaint_count_6m INT NOT NULL,
    unresolved_complaints INT NOT NULL,
    days_since_last_login INT NOT NULL,
    salary_credit_consistency NUMERIC(18,4) NULL,
    credit_utilization NUMERIC(18,4) NULL,
    emi_to_income_ratio NUMERIC(18,4) NULL,
    dormant_days INT NOT NULL,
    nps_avg_12m NUMERIC(18,4) NULL,
    campaign_response_rate NUMERIC(18,4) NULL,
    product_acquisition_velocity_6m INT NOT NULL,
    churned BOOLEAN NOT NULL,
    churn_date DATE NULL,
    churn_reason VARCHAR(100) NULL,
    PRIMARY KEY (customer_id, as_of_month, prediction_horizon_months)
);
