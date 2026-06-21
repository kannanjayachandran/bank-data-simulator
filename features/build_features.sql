-- DuckDB SQL feature materialization for churn_feature_snapshot
-- Reads already-populated customer_churn_label and monthly tables.
-- Grain: (customer_id, as_of_month, prediction_horizon_months)

DELETE FROM churn_feature_snapshot;

INSERT INTO churn_feature_snapshot (
    customer_id,
    as_of_month,
    prediction_horizon_months,
    tenure_months,
    products_count,
    balance_change_3m,
    txn_count_change_3m,
    login_count_change_6m,
    complaint_count_6m,
    unresolved_complaints,
    days_since_last_login,
    salary_credit_consistency,
    credit_utilization,
    emi_to_income_ratio,
    dormant_days,
    nps_avg_12m,
    campaign_response_rate,
    product_acquisition_velocity_6m,
    churned,
    churn_date,
    churn_reason
)
WITH spine AS (
    SELECT DISTINCT customer_id, as_of_month
    FROM customer_churn_label
    WHERE as_of_month >= (SELECT MIN(snapshot_month) + INTERVAL 6 MONTH FROM product_holdings_monthly)
),
monthly_balances AS (
    SELECT 
        acc.customer_id,
        snap.snapshot_month,
        AVG(snap.current_balance) AS avg_bal
    FROM account_monthly_snapshot snap
    JOIN account_master acc ON snap.account_id = acc.account_id
    GROUP BY acc.customer_id, snap.snapshot_month
),
recent_bal AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        AVG(mb.avg_bal) AS avg_bal_recent
    FROM spine s
    LEFT JOIN monthly_balances mb ON s.customer_id = mb.customer_id
        AND mb.snapshot_month >= CAST(s.as_of_month - INTERVAL 3 MONTH AS DATE)
        AND mb.snapshot_month < s.as_of_month
    GROUP BY s.customer_id, s.as_of_month
),
prior_bal AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        AVG(mb.avg_bal) AS avg_bal_prior
    FROM spine s
    LEFT JOIN monthly_balances mb ON s.customer_id = mb.customer_id
        AND mb.snapshot_month >= CAST(s.as_of_month - INTERVAL 6 MONTH AS DATE)
        AND mb.snapshot_month < CAST(s.as_of_month - INTERVAL 3 MONTH AS DATE)
    GROUP BY s.customer_id, s.as_of_month
),
recent_txns AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        AVG(act.debit_txn_count + act.credit_txn_count) AS avg_txn_recent
    FROM spine s
    LEFT JOIN customer_monthly_activity act ON s.customer_id = act.customer_id
        AND act.snapshot_month >= CAST(s.as_of_month - INTERVAL 3 MONTH AS DATE)
        AND act.snapshot_month < s.as_of_month
    GROUP BY s.customer_id, s.as_of_month
),
prior_txns AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        AVG(act.debit_txn_count + act.credit_txn_count) AS avg_txn_prior
    FROM spine s
    LEFT JOIN customer_monthly_activity act ON s.customer_id = act.customer_id
        AND act.snapshot_month >= CAST(s.as_of_month - INTERVAL 6 MONTH AS DATE)
        AND act.snapshot_month < CAST(s.as_of_month - INTERVAL 3 MONTH AS DATE)
    GROUP BY s.customer_id, s.as_of_month
),
recent_logins AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        AVG(act.login_count) AS avg_login_recent
    FROM spine s
    LEFT JOIN customer_monthly_activity act ON s.customer_id = act.customer_id
        AND act.snapshot_month >= CAST(s.as_of_month - INTERVAL 6 MONTH AS DATE)
        AND act.snapshot_month < s.as_of_month
    GROUP BY s.customer_id, s.as_of_month
),
prior_logins AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        AVG(act.login_count) AS avg_login_prior
    FROM spine s
    LEFT JOIN customer_monthly_activity act ON s.customer_id = act.customer_id
        AND act.snapshot_month >= CAST(s.as_of_month - INTERVAL 12 MONTH AS DATE)
        AND act.snapshot_month < CAST(s.as_of_month - INTERVAL 6 MONTH AS DATE)
    GROUP BY s.customer_id, s.as_of_month
),
complaints_6m AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        COUNT(c.complaint_id) AS complaint_count_6m
    FROM spine s
    LEFT JOIN customer_complaints c ON s.customer_id = c.customer_id
        AND c.complaint_month >= CAST(s.as_of_month - INTERVAL 6 MONTH AS DATE)
        AND c.complaint_month < s.as_of_month
    GROUP BY s.customer_id, s.as_of_month
),
unresolved_comps AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        COUNT(c.complaint_id) AS unresolved_complaints
    FROM spine s
    LEFT JOIN customer_complaints c ON s.customer_id = c.customer_id
        AND c.complaint_month < s.as_of_month
        AND c.resolved_flag = FALSE
    GROUP BY s.customer_id, s.as_of_month
),
days_since_login AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        COALESCE(
            date_diff('day', CAST(d.last_login_date AS DATE), s.as_of_month),
            180
        ) AS days_since_last_login
    FROM spine s
    LEFT JOIN digital_engagement_monthly d ON s.customer_id = d.customer_id
        AND d.snapshot_month = CAST(s.as_of_month - INTERVAL 1 MONTH AS DATE)
),
salary_consistency AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        COUNT(DISTINCT snap.snapshot_month) / 6.0 AS salary_credit_consistency
    FROM spine s
    LEFT JOIN account_master acc ON s.customer_id = acc.customer_id
    LEFT JOIN account_monthly_snapshot snap ON acc.account_id = snap.account_id
        AND snap.snapshot_month >= CAST(s.as_of_month - INTERVAL 6 MONTH AS DATE)
        AND snap.snapshot_month < s.as_of_month
        AND snap.salary_credit_amount > 0
    GROUP BY s.customer_id, s.as_of_month
),
credit_util AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        AVG(snap.utilization_rate) AS credit_utilization
    FROM spine s
    LEFT JOIN card_portfolio card ON s.customer_id = card.customer_id
    LEFT JOIN card_monthly_snapshot snap ON card.card_id = snap.card_id
        AND snap.snapshot_month = CAST(s.as_of_month - INTERVAL 1 MONTH AS DATE)
    GROUP BY s.customer_id, s.as_of_month
),
emi_ratio AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        COALESCE(le.total_emi, 0.0) / (c.annual_income / 12.0) AS emi_to_income_ratio
    FROM spine s
    JOIN customer_master c ON s.customer_id = c.customer_id
    LEFT JOIN (
        SELECT 
            lm.customer_id,
            snap.snapshot_month,
            SUM(snap.emi_amount) AS total_emi
        FROM loan_monthly_snapshot snap
        JOIN loan_master lm ON snap.loan_id = lm.loan_id
        GROUP BY lm.customer_id, snap.snapshot_month
    ) le ON s.customer_id = le.customer_id
        AND le.snapshot_month = CAST(s.as_of_month - INTERVAL 1 MONTH AS DATE)
),
dorm_days AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        COALESCE(act.days_since_last_txn, 180) AS dormant_days
    FROM spine s
    LEFT JOIN customer_monthly_activity act ON s.customer_id = act.customer_id
        AND act.snapshot_month = CAST(s.as_of_month - INTERVAL 1 MONTH AS DATE)
),
nps_avg AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        AVG(f.nps_score) AS nps_avg_12m
    FROM spine s
    LEFT JOIN customer_feedback f ON s.customer_id = f.customer_id
        AND f.feedback_month >= CAST(s.as_of_month - INTERVAL 12 MONTH AS DATE)
        AND f.feedback_month < s.as_of_month
        AND f.nps_score IS NOT NULL
    GROUP BY s.customer_id, s.as_of_month
),
campaign_rate AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        SUM(d.campaigns_responded) * 1.0 / NULLIF(SUM(d.campaigns_received), 0) AS campaign_response_rate
    FROM spine s
    LEFT JOIN digital_engagement_monthly d ON s.customer_id = d.customer_id
        AND d.snapshot_month >= CAST(s.as_of_month - INTERVAL 12 MONTH AS DATE)
        AND d.snapshot_month < s.as_of_month
    GROUP BY s.customer_id, s.as_of_month
),
prod_velocity AS (
    SELECT 
        s.customer_id,
        s.as_of_month,
        GREATEST(
            0,
            COALESCE(p1.products_count, 0) - COALESCE(p6.products_count, 0)
        ) AS product_acquisition_velocity_6m
    FROM spine s
    LEFT JOIN product_holdings_monthly p1 ON s.customer_id = p1.customer_id
        AND p1.snapshot_month = CAST(s.as_of_month - INTERVAL 1 MONTH AS DATE)
    LEFT JOIN product_holdings_monthly p6 ON s.customer_id = p6.customer_id
        AND p6.snapshot_month = CAST(s.as_of_month - INTERVAL 6 MONTH AS DATE)
)
SELECT 
    lbl.customer_id,
    lbl.as_of_month,
    lbl.prediction_horizon_months,
    -- tenure_months: (year(as_of) - year(customer_since))*12 + month(as_of) - month(customer_since)
    (EXTRACT(year FROM lbl.as_of_month) - EXTRACT(year FROM c.customer_since)) * 12 + 
    (EXTRACT(month FROM lbl.as_of_month) - EXTRACT(month FROM c.customer_since)) AS tenure_months,
    COALESCE(p1.products_count, 1) AS products_count,
    -- balance_change_3m
    -- Formula: avg_balance([M-3, M)) / nullif(avg_balance([M-6, M-3)), 0) - 1.0
    ROUND(CASE WHEN pb.avg_bal_prior > 0 THEN (rb.avg_bal_recent - pb.avg_bal_prior) / pb.avg_bal_prior ELSE 0.0 END, 4) AS balance_change_3m,
    -- txn_count_change_3m
    -- Formula: avg_txns([M-3, M)) / nullif(avg_txns([M-6, M-3)), 0) - 1.0
    ROUND(CASE WHEN pt.avg_txn_prior > 0 THEN (rt.avg_txn_recent - pt.avg_txn_prior) / pt.avg_txn_prior ELSE 0.0 END, 4) AS txn_count_change_3m,
    -- login_count_change_6m
    -- Formula: avg_logins([M-6, M)) / nullif(avg_logins([M-12, M-6)), 0) - 1.0
    ROUND(CASE WHEN pll.avg_login_prior > 0 THEN (rl.avg_login_recent - pll.avg_login_prior) / pll.avg_login_prior ELSE 0.0 END, 4) AS login_count_change_6m,
    COALESCE(c6.complaint_count_6m, 0) AS complaint_count_6m,
    COALESCE(uc.unresolved_complaints, 0) AS unresolved_complaints,
    dsl.days_since_last_login,
    ROUND(COALESCE(sc.salary_credit_consistency, 0.0), 4) AS salary_credit_consistency,
    ROUND(COALESCE(cu.credit_utilization, 0.0), 4) AS credit_utilization,
    ROUND(COALESCE(er.emi_to_income_ratio, 0.0), 4) AS emi_to_income_ratio,
    dd.dormant_days,
    ROUND(COALESCE(nps.nps_avg_12m, 8.0), 4) AS nps_avg_12m,
    ROUND(COALESCE(cr.campaign_response_rate, 0.0), 4) AS campaign_response_rate,
    -- product_acquisition_velocity_6m
    -- Formula: products_count at snapshot_month = M-1 minus products_count at snapshot_month = M-6, floored at 0
    pv.product_acquisition_velocity_6m,
    lbl.churned,
    lbl.churn_date,
    lbl.churn_reason
FROM customer_churn_label lbl
JOIN spine s ON lbl.customer_id = s.customer_id AND lbl.as_of_month = s.as_of_month
JOIN customer_master c ON lbl.customer_id = c.customer_id
LEFT JOIN product_holdings_monthly p1 ON lbl.customer_id = p1.customer_id
    AND p1.snapshot_month = CAST(lbl.as_of_month - INTERVAL 1 MONTH AS DATE)
LEFT JOIN recent_bal rb ON lbl.customer_id = rb.customer_id AND lbl.as_of_month = rb.as_of_month
LEFT JOIN prior_bal pb ON lbl.customer_id = pb.customer_id AND lbl.as_of_month = pb.as_of_month
LEFT JOIN recent_txns rt ON lbl.customer_id = rt.customer_id AND lbl.as_of_month = rt.as_of_month
LEFT JOIN prior_txns pt ON lbl.customer_id = pt.customer_id AND lbl.as_of_month = pt.as_of_month
LEFT JOIN recent_logins rl ON lbl.customer_id = rl.customer_id AND lbl.as_of_month = rl.as_of_month
LEFT JOIN prior_logins pll ON lbl.customer_id = pll.customer_id AND lbl.as_of_month = pll.as_of_month
LEFT JOIN complaints_6m c6 ON lbl.customer_id = c6.customer_id AND lbl.as_of_month = c6.as_of_month
LEFT JOIN unresolved_comps uc ON lbl.customer_id = uc.customer_id AND lbl.as_of_month = uc.as_of_month
LEFT JOIN days_since_login dsl ON lbl.customer_id = dsl.customer_id AND lbl.as_of_month = dsl.as_of_month
LEFT JOIN salary_consistency sc ON lbl.customer_id = sc.customer_id AND lbl.as_of_month = sc.as_of_month
LEFT JOIN credit_util cu ON lbl.customer_id = cu.customer_id AND lbl.as_of_month = cu.as_of_month
LEFT JOIN emi_ratio er ON lbl.customer_id = er.customer_id AND lbl.as_of_month = er.as_of_month
LEFT JOIN dorm_days dd ON lbl.customer_id = dd.customer_id AND lbl.as_of_month = dd.as_of_month
LEFT JOIN nps_avg nps ON lbl.customer_id = nps.customer_id AND lbl.as_of_month = nps.as_of_month
LEFT JOIN campaign_rate cr ON lbl.customer_id = cr.customer_id AND lbl.as_of_month = cr.as_of_month
LEFT JOIN prod_velocity pv ON lbl.customer_id = pv.customer_id AND lbl.as_of_month = pv.as_of_month;
