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
    print(f"  - Final Account Balance Distribution KS:  {ks_balance:.4f} (Target <= 0.05)")
    print(f"  - Transaction Count Distribution KS:     {ks_txns:.4f} (Target <= 0.05)")

    ks_ok = (ks_income <= 0.05) and (ks_balance <= 0.05) and (ks_txns <= 0.05)
    if ks_ok:
        print("✔ Kolmogorov-Smirnov tests: PASSED")
    else:
        print("❌ Kolmogorov-Smirnov tests: FAILED")

    print("\n======================================================================")
    if overall_ok and personas_ok and ks_ok:
        print("OVERALL RESULT: PARITY SUCCESSFULLY VERIFIED ✔")
        sys.exit(0)
    else:
        print("OVERALL RESULT: PARITY VERIFICATION FAILED ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
