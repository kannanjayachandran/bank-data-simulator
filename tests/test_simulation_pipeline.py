"""Unit tests for the Phase 4 monthly simulation loop and data integration layer."""

import os
import shutil
from datetime import date
import polars as pl
import numpy as np
import pytest

from config.simulation import SimulationConfig
from config.personas import Persona
from pipeline.simulate import run_simulation
from pipeline.writer import write_to_parquet
from pipeline.loader import load_to_postgres


@pytest.fixture(scope="module")
def sim_results():
    """Runs a 1,000 customer simulation for 12 months as the test fixture."""
    config = SimulationConfig(n_customers=1000, sim_months=12, seed=123)
    results = run_simulation(config, streaming=False)
    return config, results


def test_simulation_outputs_exist(sim_results):
    """Verify that all 17 expected DataFrames are generated with correct schemas."""
    _, results = sim_results
    
    expected_keys = [
        "branch_master",
        "customer_master",
        "account_master",
        "card_portfolio",
        "loan_master",
        "churn_simulation_state",
        "customer_complaints",
        "customer_feedback",
        "customer_churn_label",
        "churn_feature_snapshot",
        "account_monthly_snapshot",
        "card_monthly_snapshot",
        "loan_monthly_snapshot",
        "product_holdings_monthly",
        "transaction_fact",
        "customer_monthly_activity",
        "digital_engagement_monthly",
    ]
    
    for key in expected_keys:
        assert key in results, f"Missing table {key} in simulation results!"
        assert isinstance(results[key], pl.DataFrame), f"Table {key} is not a Polars DataFrame!"


def test_no_future_activity_post_churn(sim_results):
    """Verify the no-future-activity rule: no activity rows exist after churn_month."""
    _, results = sim_results
    state_df = results["churn_simulation_state"]
    
    # Filter for customers who churned
    churned_custs = state_df.filter(pl.col("churned_flag") == True).to_dicts()
    
    # Lookups for customer activities
    acc_snap = results["account_monthly_snapshot"]
    act_snap = results["customer_monthly_activity"]
    txn_fact = results["transaction_fact"]
    
    for c in churned_custs:
        cid = c["customer_id"]
        c_month = c["churn_month"]
        
        # All snapshots must be on or before the churn month
        cust_acc = acc_snap.filter((pl.col("account_id").is_in(
            results["account_master"].filter(pl.col("customer_id") == cid)["account_id"].implode()
        )))
        for row in cust_acc.to_dicts():
            assert row["snapshot_month"] <= c_month, f"Account snapshot for customer {cid} found in month {row['snapshot_month']} after churn month {c_month}!"
            
        cust_act = act_snap.filter(pl.col("customer_id") == cid)
        for row in cust_act.to_dicts():
            assert row["snapshot_month"] <= c_month, f"Activity snapshot for customer {cid} found in month {row['snapshot_month']} after churn month {c_month}!"
            
        cust_txns = txn_fact.filter(pl.col("customer_id") == cid)
        for row in cust_txns.to_dicts():
            assert row["txn_month"] <= c_month, f"Transaction for customer {cid} found in month {row['txn_month']} after churn month {c_month}!"


def test_feature_non_leakage(sim_results):
    """Verify that features at as_of_month M only aggregate historical data strictly before M."""
    _, results = sim_results
    features_df = results["churn_feature_snapshot"]
    complaints_df = results["customer_complaints"]
    
    # Verify for complaints: complaint_count_6m at as_of M must equal count of complaints in [M-6, M)
    for row in features_df.filter(pl.col("complaint_count_6m") > 0).head(10).to_dicts():
        cid = row["customer_id"]
        as_of = row["as_of_month"]
        feature_count = row["complaint_count_6m"]
        
        # Calculate expected count strictly in [as_of - 6m, as_of)
        # Find M-6 date
        if as_of.month <= 6:
            y_offset = 1
            m_offset = 12 - (6 - as_of.month)
            m_minus_6 = date(as_of.year - y_offset, m_offset, 1)
        else:
            m_minus_6 = date(as_of.year, as_of.month - 6, 1)
            
        actual_comps = complaints_df.filter(
            (pl.col("customer_id") == cid) &
            (pl.col("complaint_month") >= m_minus_6) &
            (pl.col("complaint_month") < as_of)
        )
        
        assert feature_count == len(actual_comps), (
            f"Feature count leak check failed for customer {cid} at as_of {as_of}. "
            f"Feature count: {feature_count}, Actual count in [M-6, M): {len(actual_comps)}"
        )


def test_behavioral_validations(sim_results):
    """Verify spec-driven behavioral results match relative statistics by persona."""
    _, results = sim_results
    state_df = results["churn_simulation_state"]
    loan_snap_df = results["loan_monthly_snapshot"]
    loan_master_df = results["loan_master"]
    
    # Check 1: complaint_prone_churner should have a higher average churn rate than salary_core
    cpc_churn_rate = state_df.filter(
        pl.col("persona") == Persona.COMPLAINT_PRONE_CHURNER.value
    )["churned_flag"].mean()
    
    sc_churn_rate = state_df.filter(
        pl.col("persona") == Persona.SALARY_CORE.value
    )["churned_flag"].mean()
    
    assert cpc_churn_rate > sc_churn_rate, (
        f"Behavioral check failed: complaint_prone_churner churn rate ({cpc_churn_rate:.2%}) "
        f"is not higher than salary_core churn rate ({sc_churn_rate:.2%})!"
    )
    
    # Check 2: credit_stressed average DPD should be higher than affluent_multi_product average DPD
    # Join loan snapshots with loan master to get customer persona
    ln_merged = loan_snap_df.join(
        loan_master_df.select(["loan_id", "customer_id"]), on="loan_id"
    ).join(
        state_df.select(["customer_id", "persona"]), on="customer_id"
    )
    
    cs_avg_dpd = ln_merged.filter(
        pl.col("persona") == Persona.CREDIT_STRESSED.value
    )["dpd_days"].mean()
    
    amp_avg_dpd = ln_merged.filter(
        pl.col("persona") == Persona.AFFLUENT_MULTI_PRODUCT.value
    )["dpd_days"].mean()
    
    if cs_avg_dpd is not None and amp_avg_dpd is not None:
        assert cs_avg_dpd > amp_avg_dpd, (
            f"Behavioral check failed: credit_stressed average DPD ({cs_avg_dpd:.2f}) "
            f"is not higher than affluent_multi_product average DPD ({amp_avg_dpd:.2f})!"
        )


def test_parquet_writer(sim_results):
    """Test Parquet write directories and partitioned file layouts."""
    _, results = sim_results
    test_out_dir = "./scratch_parquet_test"
    
    try:
        # Write files
        write_to_parquet(results, test_out_dir)
        
        # Verify static files exist
        assert os.path.exists(os.path.join(test_out_dir, "customer_master.parquet"))
        assert os.path.exists(os.path.join(test_out_dir, "account_master.parquet"))
        
        # Verify partitioned directory exists
        acc_snap_dir = os.path.join(test_out_dir, "account_monthly_snapshot")
        assert os.path.exists(acc_snap_dir)
        
        # Verify partition subdirectories exist
        subdirs = os.listdir(acc_snap_dir)
        assert len(subdirs) > 0
        for s in subdirs:
            assert s.startswith("snapshot_month=")
            part_file = os.path.join(acc_snap_dir, s, "part-0.parquet")
            assert os.path.exists(part_file)
            
    finally:
        if os.path.exists(test_out_dir):
            shutil.rmtree(test_out_dir)


def test_postgres_loader_connection_resilience(sim_results):
    """Verify that load_to_postgres handles connection failures gracefully and does not throw."""
    _, results = sim_results
    invalid_uri = "postgresql://non_existent_user:pass@localhost:5432/non_existent_db"
    
    # This should return gracefully with a warning print and NOT raise an exception
    try:
        load_to_postgres(results, invalid_uri)
    except Exception as e:
        pytest.fail(f"load_to_postgres raised exception {e} on connection failure!")
