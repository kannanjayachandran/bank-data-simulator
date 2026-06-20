#!/usr/bin/env python
"""Standalone behavioral and schema validator for the synthetic banking database.

Exits non-zero on failure. Connects to DuckDB in read-only mode to prevent locks.
"""

import argparse
import sys
import duckdb


def log_status(name: str, passed: bool, msg: str = ""):
    status = "✔ PASSED" if passed else "✘ FAILED"
    details = f" - {msg}" if msg else ""
    print(f"[{status}] {name}{details}")


def run_validations(db_path: str) -> bool:
    print(f"Connecting to DuckDB database: {db_path} (read-only mode)\n")
    con = duckdb.connect(db_path, read_only=True)
    all_passed = True

    # 1. Schema Validation (PK/FK Constraints & Grain checks)
    print("--- 1. SCHEMA & GRAIN VALIDATION ---")
    
    # Tables to check single-column PK uniqueness and non-nullness
    single_pks = {
        "branch_master": "branch_code",
        "customer_master": "customer_id",
        "account_master": "account_id",
        "card_portfolio": "card_id",
        "loan_master": "loan_id",
        "churn_simulation_state": "customer_id"
    }
    
    for table, pk in single_pks.items():
        try:
            null_count = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {pk} IS NULL").fetchone()[0]
            dup_count = con.execute(f"SELECT COUNT(*) FROM (SELECT {pk} FROM {table} GROUP BY {pk} HAVING COUNT(*) > 1)").fetchone()[0]
            
            passed = (null_count == 0 and dup_count == 0)
            if not passed:
                all_passed = False
            log_status(
                f"PK constraint on {table}({pk})", 
                passed, 
                f"Nulls: {null_count}, Duplicates: {dup_count}"
            )
        except Exception as e:
            all_passed = False
            log_status(f"PK check on {table}", False, str(e))

    # Tables to check composite PK uniqueness
    composite_pks = {
        "account_monthly_snapshot": ["account_id", "snapshot_month"],
        "card_monthly_snapshot": ["card_id", "snapshot_month"],
        "loan_monthly_snapshot": ["loan_id", "snapshot_month"],
        "product_holdings_monthly": ["customer_id", "snapshot_month"],
        "customer_monthly_activity": ["customer_id", "snapshot_month"],
        "digital_engagement_monthly": ["customer_id", "snapshot_month"],
        "customer_churn_label": ["customer_id", "as_of_month", "prediction_horizon_months"],
        "churn_feature_snapshot": ["customer_id", "as_of_month", "prediction_horizon_months"]
    }

    for table, cols in composite_pks.items():
        try:
            cols_str = ", ".join(cols)
            null_conds = " OR ".join([f"{c} IS NULL" for c in cols])
            null_count = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {null_conds}").fetchone()[0]
            dup_count = con.execute(f"SELECT COUNT(*) FROM (SELECT {cols_str} FROM {table} GROUP BY {cols_str} HAVING COUNT(*) > 1)").fetchone()[0]
            
            passed = (null_count == 0 and dup_count == 0)
            if not passed:
                all_passed = False
            log_status(
                f"Grain constraint on {table}({cols_str})", 
                passed, 
                f"Nulls: {null_count}, Duplicates: {dup_count}"
            )
        except Exception as e:
            all_passed = False
            log_status(f"Grain check on {table}", False, str(e))

    # Foreign Key integrity check
    fk_checks = [
        ("account_master", "customer_id", "customer_master", "customer_id"),
        ("account_master", "branch_code", "branch_master", "branch_code"),
        ("account_monthly_snapshot", "account_id", "account_master", "account_id"),
        ("card_portfolio", "customer_id", "customer_master", "customer_id"),
        ("card_monthly_snapshot", "card_id", "card_portfolio", "card_id"),
        ("loan_master", "customer_id", "customer_master", "customer_id"),
        ("loan_master", "branch_code", "branch_master", "branch_code"),
        ("loan_monthly_snapshot", "loan_id", "loan_master", "loan_id"),
        ("product_holdings_monthly", "customer_id", "customer_master", "customer_id"),
        ("transaction_fact", "customer_id", "customer_master", "customer_id"),
        ("customer_monthly_activity", "customer_id", "customer_master", "customer_id"),
        ("digital_engagement_monthly", "customer_id", "customer_master", "customer_id"),
        ("customer_complaints", "customer_id", "customer_master", "customer_id"),
        ("customer_feedback", "customer_id", "customer_master", "customer_id"),
        ("churn_simulation_state", "customer_id", "customer_master", "customer_id"),
        ("customer_churn_label", "customer_id", "customer_master", "customer_id"),
        ("churn_feature_snapshot", "customer_id", "customer_master", "customer_id")
    ]

    for c_table, c_col, p_table, p_col in fk_checks:
        try:
            orphan_count = con.execute(f"""
                SELECT COUNT(*) 
                FROM {c_table} child 
                LEFT JOIN {p_table} parent ON child.{c_col} = parent.{p_col} 
                WHERE child.{c_col} IS NOT NULL AND parent.{p_col} IS NULL
            """).fetchone()[0]
            
            passed = (orphan_count == 0)
            if not passed:
                all_passed = False
            log_status(
                f"FK constraint: {c_table}({c_col}) -> {p_table}({p_col})", 
                passed, 
                f"Orphan rows: {orphan_count}"
            )
        except Exception as e:
            all_passed = False
            log_status(f"FK check on {c_table}({c_col})", False, str(e))

    # 2. Behavioral Validation
    print("\n--- 2. BEHAVIORAL STATS VALIDATION ---")
    
    # 2.1 Churn rate difference: complaint_prone_churner vs salary_core
    try:
        rates = con.execute("""
            SELECT 
                AVG(CASE WHEN persona = 'complaint_prone_churner' AND churned_flag THEN 1.0 ELSE 0.0 END) as cpc_rate,
                AVG(CASE WHEN persona = 'salary_core' AND churned_flag THEN 1.0 ELSE 0.0 END) as sc_rate
            FROM churn_simulation_state
        """).fetchone()
        cpc_rate, sc_rate = rates[0], rates[1]
        passed = (cpc_rate is not None and sc_rate is not None and cpc_rate > sc_rate)
        if not passed:
            all_passed = False
        log_status(
            "Persona churn rate relation", 
            passed, 
            f"complaint_prone_churner={cpc_rate:.2%}, salary_core={sc_rate:.2%}"
        )
    except Exception as e:
        all_passed = False
        log_status("Persona churn check", False, str(e))

    # 2.2 DPD stress differences: credit_stressed vs affluent_multi_product
    try:
        dpds = con.execute("""
            SELECT 
                AVG(CASE WHEN s.persona = 'credit_stressed' THEN snap.dpd_days ELSE NULL END) as cs_dpd,
                AVG(CASE WHEN s.persona = 'affluent_multi_product' THEN snap.dpd_days ELSE NULL END) as amp_dpd
            FROM loan_monthly_snapshot snap
            JOIN loan_master lm ON snap.loan_id = lm.loan_id
            JOIN churn_simulation_state s ON lm.customer_id = s.customer_id
        """).fetchone()
        cs_dpd, amp_dpd = dpds[0], dpds[1]
        passed = (cs_dpd is not None and amp_dpd is not None and cs_dpd > amp_dpd)
        if not passed:
            all_passed = False
        log_status(
            "DPD stress relation", 
            passed, 
            f"credit_stressed avg DPD={cs_dpd:.2f}, affluent_multi_product avg DPD={amp_dpd:.2f}"
        )
    except Exception as e:
        all_passed = False
        log_status("DPD check", False, str(e))

    # 2.3 Campaign success vs product uptake correlation
    try:
        velocities = con.execute("""
            SELECT 
                AVG(CASE WHEN campaign_response_rate > 0 THEN product_acquisition_velocity_6m ELSE NULL END) as high_resp,
                AVG(CASE WHEN campaign_response_rate = 0 THEN product_acquisition_velocity_6m ELSE NULL END) as zero_resp
            FROM churn_feature_snapshot
        """).fetchone()
        high_resp, zero_resp = velocities[0], velocities[1]
        passed = (high_resp is not None and zero_resp is not None and high_resp > zero_resp)
        if not passed:
            all_passed = False
        log_status(
            "Campaign uptake correlation", 
            passed, 
            f"avg velocity for response_rate > 0: {high_resp:.4f}, zero response_rate: {zero_resp:.4f}"
        )
    except Exception as e:
        all_passed = False
        log_status("Campaign uptake correlation check", False, str(e))

    # 2.4 Low sensitivity slope (overall lower churn rate)
    try:
        sens = con.execute("""
            SELECT 
                AVG(CASE WHEN low_sensitivity_segment AND churned_flag THEN 1.0 WHEN low_sensitivity_segment THEN 0.0 ELSE NULL END) as low_sens,
                AVG(CASE WHEN NOT low_sensitivity_segment AND churned_flag THEN 1.0 WHEN NOT low_sensitivity_segment THEN 0.0 ELSE NULL END) as normal
            FROM churn_simulation_state
            """).fetchone()
        low_sens, normal = sens[0], sens[1]
        passed = (low_sens is not None and normal is not None and low_sens < normal)
        if not passed:
            all_passed = False
        log_status(
            "Low-sensitivity segment slope", 
            passed, 
            f"low_sensitivity={low_sens:.2%}, normal={normal:.2%}"
        )
    except Exception as e:
        all_passed = False
        log_status("Low-sensitivity check", False, str(e))

    # 3. Leakage Checks
    print("\n--- 3. LEAKAGE CHECKS ---")
    
    # 3.1 Feature snapshot date boundaries (no future label leakage)
    try:
        future_leak = con.execute("""
            SELECT COUNT(*) 
            FROM churn_feature_snapshot 
            WHERE churned = TRUE AND as_of_month >= churn_date
        """).fetchone()[0]
        passed = (future_leak == 0)
        if not passed:
            all_passed = False
        log_status(
            "Label leakage boundaries (as_of_month < churn_date)", 
            passed, 
            f"Violations count: {future_leak}"
        )
    except Exception as e:
        all_passed = False
        log_status("Label leakage check", False, str(e))

    # 3.2 Complaint feature window strictness (feature-level leakage check / spot-check)
    try:
        # Check if complaint_count_6m is exactly equal to complaints filed strictly in [M-6, M)
        violations = con.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT f.customer_id, f.as_of_month, f.complaint_count_6m,
                       (
                           SELECT COUNT(*) 
                           FROM customer_complaints c 
                           WHERE c.customer_id = f.customer_id 
                             AND c.complaint_month >= CAST(f.as_of_month - INTERVAL 6 MONTH AS DATE)
                             AND c.complaint_month < f.as_of_month
                       ) AS expected
                FROM churn_feature_snapshot f
            )
            WHERE complaint_count_6m != expected
        """).fetchone()[0]
        passed = (violations == 0)
        if not passed:
            all_passed = False
        log_status(
            "Complaint feature window strictness ([M-6, M))", 
            passed, 
            f"Violations count: {violations}"
        )
    except Exception as e:
        all_passed = False
        log_status("Complaint leakage spot-check", False, str(e))

    # 4. No Future Activity post-churn check
    print("\n--- 4. POST-CHURN INACTIVITY CHECKS ---")
    monthly_tables_date_col = {
        "account_monthly_snapshot": "snapshot_month",
        "card_monthly_snapshot": "snapshot_month",
        "loan_monthly_snapshot": "snapshot_month",
        "product_holdings_monthly": "snapshot_month",
        "customer_monthly_activity": "snapshot_month",
        "digital_engagement_monthly": "snapshot_month",
        "customer_complaints": "complaint_month",
        "customer_feedback": "feedback_month",
        "customer_churn_label": "as_of_month",
        "churn_feature_snapshot": "as_of_month",
    }
    
    # We join with loan_master or account_master for account/card/loan snapshots to get customer_id
    for table, date_col in monthly_tables_date_col.items():
        try:
            if table in ["account_monthly_snapshot", "transaction_fact"]:
                # Join with account_master
                query = f"""
                    SELECT COUNT(*) 
                    FROM {table} snap
                    JOIN account_master acc ON snap.account_id = acc.account_id
                    JOIN churn_simulation_state state ON acc.customer_id = state.customer_id
                    WHERE state.churned_flag = TRUE 
                      AND {"snap.txn_month" if table == "transaction_fact" else "snap.snapshot_month"} > state.churn_month
                """
            elif table == "card_monthly_snapshot":
                # Join with card_portfolio
                query = """
                    SELECT COUNT(*) 
                    FROM card_monthly_snapshot snap
                    JOIN card_portfolio card ON snap.card_id = card.card_id
                    JOIN churn_simulation_state state ON card.customer_id = state.customer_id
                    WHERE state.churned_flag = TRUE AND snap.snapshot_month > state.churn_month
                """
            elif table == "loan_monthly_snapshot":
                # Join with loan_master
                query = """
                    SELECT COUNT(*) 
                    FROM loan_monthly_snapshot snap
                    JOIN loan_master lm ON snap.loan_id = lm.loan_id
                    JOIN churn_simulation_state state ON lm.customer_id = state.customer_id
                    WHERE state.churned_flag = TRUE AND snap.snapshot_month > state.churn_month
                """
            else:
                query = f"""
                    SELECT COUNT(*) 
                    FROM {table} snap
                    JOIN churn_simulation_state state ON snap.customer_id = state.customer_id
                    WHERE state.churned_flag = TRUE AND snap.{date_col} >= state.churn_month
                """
                # Note: customer_churn_label/churn_feature_snapshot use as_of_month. 
                # If they churned in month C, as_of_month must be strictly less than C, so as_of_month >= C is a violation.
                # For complaints/activity/snapshots, it should be strictly greater than churn_month (since they can have activity in their churn month).
                if table not in ["customer_churn_label", "churn_feature_snapshot"]:
                    query = query.replace(">=", ">")

            violations = con.execute(query).fetchone()[0]
            passed = (violations == 0)
            if not passed:
                all_passed = False
            log_status(
                f"No-future-activity in {table}", 
                passed, 
                f"Violations count: {violations}"
            )
        except Exception as e:
            all_passed = False
            log_status(f"No-future-activity check on {table}", False, str(e))

    con.close()
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Standalone behavioral database validator.")
    parser.add_argument("--db", required=True, help="Path to DuckDB database file.")
    args = parser.parse_args()

    success = run_validations(args.db)
    print("\n=========================================")
    if success:
        print("✔ ALL DB VALIDATION CHECKS PASSED.")
        sys.exit(0)
    else:
        print("✘ DB VALIDATION CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
