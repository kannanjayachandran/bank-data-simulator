import os
import sys
import polars as pl
import numpy as np


def ks_statistic(data1, data2):
    """Calculates the two-sample Kolmogorov-Smirnov statistic."""
    if len(data1) == 0 or len(data2) == 0:
        return 1.0
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = len(data1)
    n2 = len(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side="right") / n1
    cdf2 = np.searchsorted(data2, data_all, side="right") / n2
    return float(np.max(np.abs(cdf1 - cdf2)))


def check_fk_integrity(data_dir: str) -> bool:
    """Verifies that foreign key relationships are completely valid (no orphan child rows)."""
    print(f"\nChecking Foreign Key (FK) Integrity for: {data_dir}")
    try:
        # Load tables
        cust = pl.read_parquet(os.path.join(data_dir, "customer_master.parquet"))
        cust_ids = set(cust["customer_id"].to_list())
        
        def assert_fk(child_df, child_col, parent_set, name):
            if child_df.is_empty():
                return
            child_vals = child_df[child_col].to_list()
            invalid = [v for v in child_vals if v not in parent_set]
            if invalid:
                print(f"  ❌ FK Violation: {name}.{child_col} has {len(invalid)} values not in parent! (e.g. {invalid[:5]})")
                raise AssertionError(f"FK violation on {name}.{child_col}")
        
        # account_master
        acc = pl.read_parquet(os.path.join(data_dir, "account_master.parquet"))
        assert_fk(acc, "customer_id", cust_ids, "account_master")
        acc_ids = set(acc["account_id"].to_list())
        
        # account_monthly_snapshot
        acc_snap = pl.read_parquet(os.path.join(data_dir, "account_monthly_snapshot/**/*.parquet"))
        assert_fk(acc_snap, "account_id", acc_ids, "account_monthly_snapshot")
        
        # card_portfolio
        card = pl.read_parquet(os.path.join(data_dir, "card_portfolio.parquet"))
        assert_fk(card, "customer_id", cust_ids, "card_portfolio")
        card_ids = set(card["card_id"].to_list())
        
        # card_monthly_snapshot
        card_snap = pl.read_parquet(os.path.join(data_dir, "card_monthly_snapshot/**/*.parquet"))
        assert_fk(card_snap, "card_id", card_ids, "card_monthly_snapshot")
        
        # loan_master
        loan = pl.read_parquet(os.path.join(data_dir, "loan_master.parquet"))
        assert_fk(loan, "customer_id", cust_ids, "loan_master")
        loan_ids = set(loan["loan_id"].to_list())
        
        # loan_monthly_snapshot
        loan_snap = pl.read_parquet(os.path.join(data_dir, "loan_monthly_snapshot/**/*.parquet"))
        assert_fk(loan_snap, "loan_id", loan_ids, "loan_monthly_snapshot")
        
        # product_holdings_monthly
        holdings = pl.read_parquet(os.path.join(data_dir, "product_holdings_monthly/**/*.parquet"))
        assert_fk(holdings, "customer_id", cust_ids, "product_holdings_monthly")
        
        # transaction_fact
        txn = pl.read_parquet(os.path.join(data_dir, "transaction_fact/**/*.parquet"))
        assert_fk(txn, "customer_id", cust_ids, "transaction_fact")
        assert_fk(txn, "account_id", acc_ids, "transaction_fact")
        
        # customer_monthly_activity
        activity = pl.read_parquet(os.path.join(data_dir, "customer_monthly_activity/**/*.parquet"))
        assert_fk(activity, "customer_id", cust_ids, "customer_monthly_activity")
        
        # digital_engagement_monthly
        digital = pl.read_parquet(os.path.join(data_dir, "digital_engagement_monthly/**/*.parquet"))
        assert_fk(digital, "customer_id", cust_ids, "digital_engagement_monthly")
        
        # customer_complaints
        comp = pl.read_parquet(os.path.join(data_dir, "customer_complaints/**/*.parquet"))
        assert_fk(comp, "customer_id", cust_ids, "customer_complaints")
        
        # customer_feedback
        feed = pl.read_parquet(os.path.join(data_dir, "customer_feedback/**/*.parquet"))
        assert_fk(feed, "customer_id", cust_ids, "customer_feedback")
        
        # churn_simulation_state
        state = pl.read_parquet(os.path.join(data_dir, "churn_simulation_state.parquet"))
        assert_fk(state, "customer_id", cust_ids, "churn_simulation_state")
        
        # customer_churn_label
        label = pl.read_parquet(os.path.join(data_dir, "customer_churn_label/**/*.parquet"))
        assert_fk(label, "customer_id", cust_ids, "customer_churn_label")
        
        # churn_feature_snapshot
        feat = pl.read_parquet(os.path.join(data_dir, "churn_feature_snapshot/**/*.parquet"))
        assert_fk(feat, "customer_id", cust_ids, "churn_feature_snapshot")
        
        print("  ✔ FK Integrity: PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FK Integrity check failed: {e}")
        return False


def main():
    baseline_dir = "./data/baseline"
    optimized_dir = "./data/optimized"

    if not os.path.exists(baseline_dir):
        print(f"Error: Baseline directory '{baseline_dir}' does not exist.")
        sys.exit(1)
    if not os.path.exists(optimized_dir):
        print(f"Error: Optimized directory '{optimized_dir}' does not exist. Run the optimized simulation first.")
        sys.exit(1)

    print("======================================================================")
    print("VERIFYING STATISTICAL AND DISTRIBUTIONAL PARITY")
    print("======================================================================")

    # 0. Check FK Integrity
    fk_base_ok = check_fk_integrity(baseline_dir)
    fk_opt_ok = check_fk_integrity(optimized_dir)
    fk_ok = fk_base_ok and fk_opt_ok

    # 1. Load Data
    cust_base = pl.read_parquet(os.path.join(baseline_dir, "customer_master.parquet"))
    cust_opt = pl.read_parquet(os.path.join(optimized_dir, "customer_master.parquet"))

    sim_base = pl.read_parquet(os.path.join(baseline_dir, "churn_simulation_state.parquet"))
    sim_opt = pl.read_parquet(os.path.join(optimized_dir, "churn_simulation_state.parquet"))

    label_base = pl.read_parquet(os.path.join(baseline_dir, "customer_churn_label/**/*.parquet"))
    label_opt = pl.read_parquet(os.path.join(optimized_dir, "customer_churn_label/**/*.parquet"))

    # Join churn_simulation_state to get persona
    label_base = label_base.join(sim_base.select(["customer_id", "persona"]), on="customer_id")
    label_opt = label_opt.join(sim_opt.select(["customer_id", "persona"]), on="customer_id")

    # 2. Check Overall Churn Rate
    churn_rate_base = label_base.select(pl.col("churned").mean()).item()
    churn_rate_opt = label_opt.select(pl.col("churned").mean()).item()
    
    diff_overall = abs(churn_rate_base - churn_rate_opt)
    print(f"Overall Churn Rate (Baseline):  {churn_rate_base:.4%}")
    print(f"Overall Churn Rate (Optimized): {churn_rate_opt:.4%}")
    print(f"Difference:                     {diff_overall:.4%}")
    
    overall_ok = diff_overall <= 0.01
    if overall_ok:
        print("✔ Overall Churn Rate check: PASSED (<= 1%)")
    else:
        print("❌ Overall Churn Rate check: FAILED (> 1%)")

    # 3. Check Persona-level Churn Rates
    personas_base = label_base.group_by("persona").agg(pl.col("churned").mean().alias("churn_rate")).sort("persona")
    personas_opt = label_opt.group_by("persona").agg(pl.col("churned").mean().alias("churn_rate")).sort("persona")
    
    persona_diffs = []
    print("\nPersona-level Churn Rates:")
    for row_b in personas_base.iter_rows(named=True):
        persona = row_b["persona"]
        rate_b = row_b["churn_rate"]
        # Find matching row in opt
        row_o = personas_opt.filter(pl.col("persona") == persona)
        if row_o.is_empty():
            rate_o = 0.0
        else:
            rate_o = row_o["churn_rate"][0]
        
        diff = abs(rate_b - rate_o)
        persona_diffs.append(diff)
        status = "✔" if diff <= 0.02 else "❌"
        print(f"  - {persona:25}: Baseline={rate_b:.2%}, Optimized={rate_o:.2%}, Diff={diff:.2%} {status}")

    personas_ok = all(d <= 0.02 for d in persona_diffs)
    if personas_ok:
        print("✔ Persona Churn Rate checks: PASSED (all <= 2%)")
    else:
        print("❌ Persona Churn Rate checks: FAILED (some > 2%)")

    # 4. Check Distributions (Kolmogorov-Smirnov Test)
    # Income
    income_base = cust_base["annual_income"].to_numpy()
    income_opt = cust_opt["annual_income"].to_numpy()
    ks_income = ks_statistic(income_base, income_opt)
    
    # Final balances
    acct_base = pl.read_parquet(os.path.join(baseline_dir, "account_monthly_snapshot/**/*.parquet"))
    acct_opt = pl.read_parquet(os.path.join(optimized_dir, "account_monthly_snapshot/**/*.parquet"))
    
    # Let's check the last month's balances (sim_end balance)
    max_month_base = acct_base["snapshot_month"].max()
    max_month_opt = acct_opt["snapshot_month"].max()
    
    bal_base = acct_base.filter(pl.col("snapshot_month") == max_month_base)["current_balance"].to_numpy()
    bal_opt = acct_opt.filter(pl.col("snapshot_month") == max_month_opt)["current_balance"].to_numpy()
    ks_balance = ks_statistic(bal_base, bal_opt)

    # Transaction count
    txn_base_df = pl.read_parquet(os.path.join(baseline_dir, "transaction_fact/**/*.parquet"))
    txn_opt_df = pl.read_parquet(os.path.join(optimized_dir, "transaction_fact/**/*.parquet"))
    
    txns_per_cust_base = txn_base_df.group_by("customer_id").len()["len"].to_numpy()
    txns_per_cust_opt = txn_opt_df.group_by("customer_id").len()["len"].to_numpy()
    ks_txns = ks_statistic(txns_per_cust_base, txns_per_cust_opt)

    print("\nKolmogorov-Smirnov Distribution Checks:")
    print(f"  - Annual Income Distribution KS Stat:    {ks_income:.4f} (Target <= 0.05)")
    print(f"  - Final Account Balance Distribution KS:  {ks_balance:.4f} (Target <= 0.07)")
    print(f"  - Transaction Count Distribution KS:     {ks_txns:.4f} (Target <= 0.05)")

    ks_ok = (ks_income <= 0.05) and (ks_balance <= 0.07) and (ks_txns <= 0.05)
    if ks_ok:
        print("✔ Kolmogorov-Smirnov tests: PASSED")
    else:
        print("❌ Kolmogorov-Smirnov tests: FAILED")

    print("\n======================================================================")
    if overall_ok and personas_ok and ks_ok and fk_ok:
        print("OVERALL RESULT: PARITY SUCCESSFULLY VERIFIED ✔")
        sys.exit(0)
    else:
        print("OVERALL RESULT: PARITY VERIFICATION FAILED ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
