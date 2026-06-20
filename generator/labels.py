"""Derived labels and features generator for the synthetic retail banking universe.

Calculates churn labels for prediction horizons (1, 3, 6, 12 months) and compiles
historical feature snapshots for training, ensuring no future data leakage.
"""

from datetime import date
from typing import List, Dict, Any, Optional
import numpy as np
import polars as pl

from config.simulation import SimulationConfig


def generate_churn_labels(
    churn_simulation_state: pl.DataFrame,
    config: SimulationConfig,
) -> pl.DataFrame:
    """Derives customer_churn_label rows post-simulation.

    Grain: (customer_id, as_of_month, prediction_horizon_months)

    Args:
        churn_simulation_state: The final churn simulation state table.
        config: Simulation configuration.

    Returns:
        pl.DataFrame: Churn labels.
    """
    horizons = [1, 3, 6, 12]
    
    # Calculate all simulation months
    months: List[date] = []
    start = config.sim_start
    for i in range(config.sim_months):
        if start.month + i > 12:
            y_offset = (start.month + i - 1) // 12
            m_offset = (start.month + i - 1) % 12 + 1
            curr_month = date(start.year + y_offset, m_offset, 1)
        else:
            curr_month = date(start.year, start.month + i, 1)
        months.append(curr_month)

    label_rows = []

    for row in churn_simulation_state.to_dicts():
        cid = row["customer_id"]
        churn_month = row["churn_month"]  # date or None
        churn_reason = row["churn_reason"]

        for as_of in months:
            # A customer is eligible for labeling if they were active or churned in/after this month
            # Customers who churned strictly before or in this month as_of are NOT labeled for future churn
            if churn_month is not None and churn_month <= as_of:
                continue

            for horizon in horizons:
                # Target range is (as_of, as_of + horizon]
                # Calculate boundary month
                y_offset = (as_of.month + horizon - 1) // 12
                m_offset = (as_of.month + horizon - 1) % 12 + 1
                boundary = date(as_of.year + y_offset, m_offset, 1)

                churned = False
                c_date = None
                c_reason = None

                if churn_month is not None and as_of < churn_month <= boundary:
                    churned = True
                    c_date = churn_month
                    c_reason = churn_reason

                label_rows.append(
                    {
                        "customer_id": cid,
                        "as_of_month": as_of,
                        "prediction_horizon_months": horizon,
                        "churned": churned,
                        "churn_date": c_date,
                        "churn_reason": c_reason,
                    }
                )

    return pl.DataFrame(
        label_rows,
        schema={
            "customer_id": pl.Int64,
            "as_of_month": pl.Date,
            "prediction_horizon_months": pl.Int32,
            "churned": pl.Boolean,
            "churn_date": pl.Date,
            "churn_reason": pl.String,
        },
    )


def add_months(d: date, m: int) -> date:
    """Helper to add months to a date."""
    y_offset = (d.month + m - 1) // 12
    m_offset = (d.month + m - 1) % 12 + 1
    return date(d.year + y_offset, m_offset, 1)


def generate_feature_snapshots(
    customer_master: pl.DataFrame,
    account_master: pl.DataFrame,
    account_monthly_snapshot: pl.DataFrame,
    card_portfolio: pl.DataFrame,
    card_monthly_snapshot: pl.DataFrame,
    loan_master: pl.DataFrame,
    loan_monthly_snapshot: pl.DataFrame,
    product_holdings_monthly: pl.DataFrame,
    customer_monthly_activity: pl.DataFrame,
    digital_engagement_monthly: pl.DataFrame,
    customer_complaints: pl.DataFrame,
    customer_feedback: pl.DataFrame,
    customer_churn_label: pl.DataFrame,
    config: SimulationConfig,
) -> pl.DataFrame:
    """Compiles the churn_feature_snapshot table.

    Grain: (customer_id, as_of_month, prediction_horizon_months)

    Aggregates historical data strictly prior to as_of_month.

    Returns:
        pl.DataFrame: Feature snapshot table.
    """
    horizons = [1, 3, 6, 12]

    # Convert static tables to lookup dicts
    cust_since = {
        row["customer_id"]: row["customer_since"]
        for row in customer_master.select(["customer_id", "customer_since"]).to_dicts()
    }
    cust_income = {
        row["customer_id"]: float(row["annual_income"])
        for row in customer_master.select(["customer_id", "annual_income"]).to_dicts()
    }

    # Map accounts to customer
    acct_to_cust = {
        row["account_id"]: row["customer_id"]
        for row in account_master.select(["account_id", "customer_id"]).to_dicts()
    }
    
    # Pre-add customer_id to account snapshots for faster grouping
    if "customer_id" not in account_monthly_snapshot.columns:
        acct_cust_df = account_master.select(["account_id", "customer_id"])
        account_snapshots = account_monthly_snapshot.join(acct_cust_df, on="account_id", how="left")
    else:
        account_snapshots = account_monthly_snapshot

    # Pre-add customer_id to card snapshots
    if "customer_id" not in card_monthly_snapshot.columns:
        card_cust_df = card_portfolio.select(["card_id", "customer_id"])
        card_snapshots = card_monthly_snapshot.join(card_cust_df, on="card_id", how="left")
    else:
        card_snapshots = card_monthly_snapshot

    # Pre-add customer_id to loan snapshots
    if "customer_id" not in loan_monthly_snapshot.columns:
        loan_cust_df = loan_master.select(["loan_id", "customer_id"])
        loan_snapshots = loan_monthly_snapshot.join(loan_cust_df, on="loan_id", how="left")
    else:
        loan_snapshots = loan_monthly_snapshot

    # Generate the as_of_months (from sim_start + 6 to sim_end)
    months: List[date] = []
    start = config.sim_start
    for i in range(6, config.sim_months):  # require at least 6 months of history
        months.append(add_months(start, i))

    snapshot_rows = []

    # Cache label records for fast lookup
    # key: (customer_id, as_of_month, horizon) -> label_dict
    label_lookup = {}
    for lbl in customer_churn_label.to_dicts():
        key = (lbl["customer_id"], lbl["as_of_month"], lbl["prediction_horizon_months"])
        label_lookup[key] = lbl

    for as_of in months:
        # Customers eligible for feature extraction: those present in labels for this as_of
        # Which is equivalent to: active as of this month
        eligible_cids = {
            key[0] for key, lbl in label_lookup.items()
            if key[1] == as_of
        }

        if not eligible_cids:
            continue

        # Filter monthly tables for prior windows: [as_of - 6, as_of)
        m_minus_6 = add_months(as_of, -6)
        m_minus_12 = add_months(as_of, -12)

        # Pre-filter DataFrames for speed
        acct_window = account_snapshots.filter(
            (pl.col("snapshot_month") >= m_minus_6) & (pl.col("snapshot_month") < as_of)
        )
        activity_window = customer_monthly_activity.filter(
            (pl.col("snapshot_month") >= m_minus_12) & (pl.col("snapshot_month") < as_of)
        )
        digital_window = digital_engagement_monthly.filter(
            (pl.col("snapshot_month") >= m_minus_12) & (pl.col("snapshot_month") < as_of)
        )
        complaints_window = customer_complaints.filter(
            (pl.col("complaint_month") >= m_minus_6) & (pl.col("complaint_month") < as_of)
        )
        feedback_window = customer_feedback.filter(
            (pl.col("feedback_month") >= m_minus_12) & (pl.col("feedback_month") < as_of)
        )

        # Products at M-1
        m_minus_1 = add_months(as_of, -1)
        prod_m1 = product_holdings_monthly.filter(pl.col("snapshot_month") == m_minus_1)
        prod_lookup = {
            row["customer_id"]: row["products_count"]
            for row in prod_m1.select(["customer_id", "products_count"]).to_dicts()
        }

        # Products at M-6
        prod_m6 = product_holdings_monthly.filter(pl.col("snapshot_month") == m_minus_6)
        prod_m6_lookup = {
            row["customer_id"]: row["products_count"]
            for row in prod_m6.select(["customer_id", "products_count"]).to_dicts()
        }

        # Dictionaries for quick feature calculation
        # 1. Accounts balance & salary credits
        # average balance by customer and month
        bal_by_cust = {}
        sal_by_cust = {}
        for row in acct_window.to_dicts():
            cid = row["customer_id"]
            m_date = row["snapshot_month"]
            bal = float(row["current_balance"])
            sal = float(row["salary_credit_amount"])
            
            bal_by_cust.setdefault(cid, {}).setdefault(m_date, []).append(bal)
            if row["salary_credit_amount"] > 0:
                sal_by_cust.setdefault(cid, set()).add(m_date)

        # 2. Activity metrics
        act_by_cust = {}
        for row in activity_window.to_dicts():
            cid = row["customer_id"]
            m_date = row["snapshot_month"]
            txns = int(row["debit_txn_count"] + row["credit_txn_count"])
            logins = int(row["login_count"])
            dorm = int(row["days_since_last_txn"])
            act_by_cust.setdefault(cid, {})[m_date] = {"txns": txns, "logins": logins, "dormant_days": dorm}

        # 3. Digital engagement (for M-1 last login, push campaigns)
        dig_m1_lookup = {}
        campaign_sent = {}
        campaign_resp = {}
        for row in digital_window.to_dicts():
            cid = row["customer_id"]
            m_date = row["snapshot_month"]
            if m_date == m_minus_1:
                dig_m1_lookup[cid] = row
            campaign_sent.setdefault(cid, 0)
            campaign_sent[cid] += int(row["campaigns_received"])
            campaign_resp.setdefault(cid, 0)
            campaign_resp[cid] += int(row["campaigns_responded"])

        # 4. Complaints
        complaints_count = {}
        unresolved_count = {}
        for row in complaints_window.to_dicts():
            cid = row["customer_id"]
            complaints_count[cid] = complaints_count.get(cid, 0) + 1
            # Unresolved means: either not resolved yet, or resolved in/after the current month M
            if not row["resolved_flag"] or row["complaint_date"] >= as_of: # Wait, if it is resolved after as_of, it was unresolved as of M.
                unresolved_count[cid] = unresolved_count.get(cid, 0) + 1

        # Also count complaints unresolved from *before* M-6
        prior_unresolved = customer_complaints.filter(
            (pl.col("complaint_month") < m_minus_6) & 
            ((pl.col("resolved_flag") == False) | (pl.col("complaint_date") >= as_of))
        )
        for row in prior_unresolved.to_dicts():
            cid = row["customer_id"]
            unresolved_count[cid] = unresolved_count.get(cid, 0) + 1

        # 5. Feedback NPS
        nps_scores = {}
        for row in feedback_window.to_dicts():
            cid = row["customer_id"]
            if row["nps_score"] is not None:
                nps_scores.setdefault(cid, []).append(int(row["nps_score"]))

        # 6. Active loans for EMI ratio
        active_emis = {}
        # EMI in month M-1: loan must be active in M-1
        for row in loan_snapshots.filter(pl.col("snapshot_month") == m_minus_1).to_dicts():
            cid = row["customer_id"]
            active_emis[cid] = active_emis.get(cid, 0.0) + float(row["emi_amount"])

        # 7. Card utilization in M-1
        card_utils = {}
        for row in card_snapshots.filter(pl.col("snapshot_month") == m_minus_1).to_dicts():
            cid = row["customer_id"]
            card_utils.setdefault(cid, []).append(float(row["utilization_rate"]))

        for cid in eligible_cids:
            since = cust_since.get(cid)
            income = cust_income.get(cid, 120000.0)
            
            # 1. tenure_months
            tenure = (as_of.year - since.year) * 12 + (as_of.month - since.month)
            
            # 2. products_count
            p_count = prod_lookup.get(cid, 1)

            # 3. balance_change_3m
            # Formula: avg_balance([M-3, M)) / nullif(avg_balance([M-6, M-3)), 0) - 1.0
            # average balance in [M-3, M) vs [M-6, M-3)
            # average balances per month
            m_balances = bal_by_cust.get(cid, {})
            # months in recent window [M-3, M)
            recent_months = [add_months(as_of, -i) for i in range(1, 4)]
            prior_months = [add_months(as_of, -i) for i in range(4, 7)]
            
            recent_bals = []
            for rm in recent_months:
                if rm in m_balances:
                    recent_bals.append(np.mean(m_balances[rm]))
            avg_bal_recent = np.mean(recent_bals) if recent_bals else 0.0

            prior_bals = []
            for pm in prior_months:
                if pm in m_balances:
                    prior_bals.append(np.mean(m_balances[pm]))
            avg_bal_prior = np.mean(prior_bals) if prior_bals else 0.0

            if avg_bal_prior > 0:
                bal_change = (avg_bal_recent - avg_bal_prior) / avg_bal_prior
            else:
                bal_change = 0.0

            # 4. txn_count_change_3m
            m_activity = act_by_cust.get(cid, {})
            recent_txns = [m_activity[rm]["txns"] for rm in recent_months if rm in m_activity]
            avg_txn_recent = np.mean(recent_txns) if recent_txns else 0.0
            
            prior_txns = [m_activity[pm]["txns"] for pm in prior_months if pm in m_activity]
            avg_txn_prior = np.mean(prior_txns) if prior_txns else 0.0

            if avg_txn_prior > 0:
                txn_change = (avg_txn_recent - avg_txn_prior) / avg_txn_prior
            else:
                txn_change = 0.0

            # 5. login_count_change_6m (recent 6m vs prior 6m)
            recent_months_6m = [add_months(as_of, -i) for i in range(1, 7)]
            prior_months_6m = [add_months(as_of, -i) for i in range(7, 13)]
            
            recent_logins = [m_activity[rm]["logins"] for rm in recent_months_6m if rm in m_activity]
            avg_login_recent = np.mean(recent_logins) if recent_logins else 0.0
            
            prior_logins = [m_activity[pm]["logins"] for pm in prior_months_6m if pm in m_activity]
            avg_login_prior = np.mean(prior_logins) if prior_logins else 0.0

            if avg_login_prior > 0:
                login_change = (avg_login_recent - avg_login_prior) / avg_login_prior
            else:
                login_change = 0.0

            # 6. complaint_count_6m
            comp_6m = complaints_count.get(cid, 0)

            # 7. unresolved_complaints
            unres_comp = unresolved_count.get(cid, 0)

            # 8. days_since_last_login
            dig_m1 = dig_m1_lookup.get(cid)
            if dig_m1 and dig_m1["last_login_date"] is not None:
                days_login = (as_of - dig_m1["last_login_date"]).days
            else:
                days_login = 180

            # 9. salary_credit_consistency
            sal_months = sal_by_cust.get(cid, set())
            sal_consistency = len(sal_months) / 6.0

            # 10. credit_utilization
            utils = card_utils.get(cid, [])
            cred_util = np.mean(utils) if utils else 0.0

            # 11. emi_to_income_ratio
            emi_total = active_emis.get(cid, 0.0)
            emi_ratio = emi_total / (income / 12.0)

            # 12. dormant_days (days since last txn as of end of M-1)
            act_m1 = m_activity.get(m_minus_1)
            dorm_d = act_m1["dormant_days"] if act_m1 else 180

            # 13. nps_avg_12m
            nps_list = nps_scores.get(cid, [])
            nps_avg = np.mean(nps_list) if nps_list else 8.0

            # 14. campaign_response_rate
            c_sent = campaign_sent.get(cid, 0)
            c_resp = campaign_resp.get(cid, 0)
            c_rate = c_resp / c_sent if c_sent > 0 else 0.0

            # 15. product_acquisition_velocity_6m
            # Formula: products_count at snapshot_month = M-1 minus products_count at snapshot_month = M-6, floored at 0
            prod_m1_cnt = prod_lookup.get(cid, 0)
            prod_m6_cnt = prod_m6_lookup.get(cid, 0)
            prod_velocity = max(0, prod_m1_cnt - prod_m6_cnt)

            for horizon in horizons:
                lbl = label_lookup.get((cid, as_of, horizon))
                if not lbl:
                    continue

                snapshot_rows.append(
                    {
                        "customer_id": cid,
                        "as_of_month": as_of,
                        "prediction_horizon_months": horizon,
                        "tenure_months": int(tenure),
                        "products_count": int(p_count),
                        "balance_change_3m": float(round(bal_change, 4)),
                        "txn_count_change_3m": float(round(txn_change, 4)),
                        "login_count_change_6m": float(round(login_change, 4)),
                        "complaint_count_6m": int(comp_6m),
                        "unresolved_complaints": int(unres_comp),
                        "days_since_last_login": int(days_login),
                        "salary_credit_consistency": float(round(sal_consistency, 4)),
                        "credit_utilization": float(round(cred_util, 4)),
                        "emi_to_income_ratio": float(round(emi_ratio, 4)),
                        "dormant_days": int(dorm_d),
                        "nps_avg_12m": float(round(nps_avg, 4)),
                        "campaign_response_rate": float(round(c_rate, 4)),
                        "product_acquisition_velocity_6m": int(prod_velocity),
                        "churned": lbl["churned"],
                        "churn_date": lbl["churn_date"],
                        "churn_reason": lbl["churn_reason"],
                    }
                )

    return pl.DataFrame(
        snapshot_rows,
        schema={
            "customer_id": pl.Int64,
            "as_of_month": pl.Date,
            "prediction_horizon_months": pl.Int32,
            "tenure_months": pl.Int32,
            "products_count": pl.Int32,
            "balance_change_3m": pl.Float64,
            "txn_count_change_3m": pl.Float64,
            "login_count_change_6m": pl.Float64,
            "complaint_count_6m": pl.Int32,
            "unresolved_complaints": pl.Int32,
            "days_since_last_login": pl.Int32,
            "salary_credit_consistency": pl.Float64,
            "credit_utilization": pl.Float64,
            "emi_to_income_ratio": pl.Float64,
            "dormant_days": pl.Int32,
            "nps_avg_12m": pl.Float64,
            "campaign_response_rate": pl.Float64,
            "product_acquisition_velocity_6m": pl.Int32,
            "churned": pl.Boolean,
            "churn_date": pl.Date,
            "churn_reason": pl.String,
        },
    )
