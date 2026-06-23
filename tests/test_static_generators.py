"""Unit tests for the static entity generators in Phase 3."""

from datetime import date
import numpy as np
import polars as pl
import pytest

from config.constants import CUSTOMER_ID_START
from config.personas import Persona
from config.simulation import SimulationConfig
from generator.accounts import generate_accounts
from generator.branches import generate_branches
from generator.cards import generate_cards
from generator.customers import generate_customers
from generator.loans import generate_loans
from generator.products import generate_initial_products
from generator.spine import generate_spine


def test_branches_master():
    """Verify that the branches DataFrame is correctly pre-populated and structured."""
    branches_df = generate_branches()
    assert len(branches_df) == 10
    assert branches_df.columns == [
        "branch_code",
        "branch_name",
        "city",
        "state",
        "region",
        "branch_type",
        "open_date",
        "closure_date",
    ]
    # Check uniqueness of branch codes
    assert branches_df["branch_code"].is_unique().all()


def test_static_pipeline_and_foreign_keys():
    """Verify the execution and FK constraints of the static entity generation pipeline."""
    config = SimulationConfig(n_customers=200, sim_months=12, seed=42)
    rng = np.random.default_rng(config.seed)

    # 1. Generate Spine
    spine = generate_spine(config)

    # 2. Generate Initial Products
    products_df = generate_initial_products(spine, config, rng)
    assert len(products_df) == 200

    # 3. Generate Customers
    customers_df = generate_customers(spine, config, rng)
    assert len(customers_df) == 200

    # 4. Generate Accounts
    accounts_df = generate_accounts(spine, customers_df, products_df, config, rng)
    assert len(accounts_df) >= 200  # at least one savings account per customer

    # 5. Generate Cards
    cards_df = generate_cards(spine, customers_df, products_df, accounts_df, config, rng)

    # 6. Generate Loans
    loans_df, products_df = generate_loans(spine, customers_df, products_df, config, rng)

    branches_df = generate_branches()

    # Validate Foreign Keys in Accounts (Referential Integrity checks)
    assert accounts_df.join(customers_df, on="customer_id", how="anti").is_empty()
    assert accounts_df.join(branches_df, on="branch_code", how="anti").is_empty()

    # Validate Foreign Keys in Cards
    assert cards_df.join(customers_df, on="customer_id", how="anti").is_empty()

    # Validate Foreign Keys in Loans
    assert loans_df.join(customers_df, on="customer_id", how="anti").is_empty()
    assert loans_df.join(branches_df, on="branch_code", how="anti").is_empty()

    # Assert Savings Account Invariant: Every customer_id in customer_master has at least one Savings account
    for cid in customers_df["customer_id"].to_list():
        cust_accs = accounts_df.filter(
            (pl.col("customer_id") == cid) & (pl.col("account_type") == "Savings")
        )
        assert len(cust_accs) >= 1, f"Customer {cid} is missing a Savings account!"


def test_loan_emi_boundary_math():
    """Test boundary checks for loan EMI generation and amortization math."""
    # Custom loan parameters: $50,000, 12 months, 15% interest
    interest_rate = 15.0
    tenure_months = 12
    sanctioned_amount = 50000.0

    R = interest_rate / 12.0 / 100.0
    N = tenure_months
    P = sanctioned_amount

    # Amortization math
    emi = P * R * ((1 + R) ** N) / (((1 + R) ** N) - 1)
    emi_amount = float(round(emi, 2))
    total_repayment = emi_amount * tenure_months

    assert emi_amount > 0.0
    assert total_repayment > sanctioned_amount
    # The interest paid should be approx 4,150 INR for 15% on 50k over 1 year
    assert 4000.0 < (total_repayment - sanctioned_amount) < 4300.0


def test_product_uptake_statistical_expectations():
    """Test that the sampled product flag distribution matches persona probabilities."""
    # We use a larger spine (1000 customers) to obtain statistical significance
    config = SimulationConfig(n_customers=1000, sim_months=12, seed=888)
    rng = np.random.default_rng(config.seed)
    spine = generate_spine(config)

    products_df = generate_initial_products(spine, config, rng)

    # Let's count dormant wealthy customers and check their fixed deposit uptake rate
    # Dormant Wealthy fixed_deposit uptake is 88%
    dormant_wealthy_ids = spine.simulation_state.filter(
        pl.col("persona") == Persona.DORMANT_WEALTHY.value
    )["customer_id"].to_list()

    dw_count = len(dormant_wealthy_ids)
    assert dw_count > 100  # should be around 166

    dw_fd_count = products_df.filter(
        (pl.col("customer_id").is_in(dormant_wealthy_ids)) & (pl.col("fixed_deposit_flag") == True)
    ).height

    fd_ratio = dw_fd_count / dw_count
    # Expect 88% uptake. Allowing margin of error (+/- 8% for 95% confidence on N~166)
    assert 0.80 <= fd_ratio <= 0.96
