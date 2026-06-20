"""Initial product holdings generator for the synthetic retail banking universe.

Determines the products held by each customer at the start of the simulation.
"""

from typing import Optional
import numpy as np
import polars as pl

from config.personas import PERSONA_CONFIGS, Persona
from config.simulation import SimulationConfig
from generator.spine import Spine


def generate_initial_products(
    spine: Spine,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generates the initial product holdings for all customers.

    Args:
        spine: The customer spine containing personas.
        config: Simulation configuration.
        rng: Optional seeded numpy random generator.

    Returns:
        pl.DataFrame: DataFrame containing initial product holding flags for all 11 products.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    state_df = spine.simulation_state
    customer_ids = state_df["customer_id"].to_list()
    personas = state_df["persona"].to_list()

    product_keys = [
        "savings",
        "current",
        "debit_card",
        "credit_card",
        "personal_loan",
        "home_loan",
        "fixed_deposit",
        "insurance",
        "mutual_fund",
        "demat_account",
        "wealth_management",
    ]

    rows = []
    for cid, pers_name in zip(customer_ids, personas):
        p_enum = Persona(pers_name)
        p_config = PERSONA_CONFIGS[p_enum]
        probs = p_config.product_uptake_probs

        row = {"customer_id": cid, "snapshot_month": config.sim_start}

        # Sample each product flag
        for key in product_keys:
            prob = probs.get(key, 0.0)
            row[f"{key}_flag"] = rng.random() < prob

        # Savings account invariant: every customer must have a Savings account
        row["savings_flag"] = True

        # Calculate products_count excluding debit_card_flag (as debit card is portfolio-only)
        count_flags = [
            row["savings_flag"],
            row["current_flag"],
            row["credit_card_flag"],
            row["personal_loan_flag"],
            row["home_loan_flag"],
            row["fixed_deposit_flag"],
            row["insurance_flag"],
            row["mutual_fund_flag"],
            row["demat_account_flag"],
            row["wealth_management_flag"],
        ]
        row["products_count"] = sum(count_flags)

        rows.append(row)

    return pl.DataFrame(
        rows,
        schema={
            "customer_id": pl.Int64,
            "snapshot_month": pl.Date,
            "savings_flag": pl.Boolean,
            "current_flag": pl.Boolean,
            "debit_card_flag": pl.Boolean,
            "credit_card_flag": pl.Boolean,
            "personal_loan_flag": pl.Boolean,
            "home_loan_flag": pl.Boolean,
            "fixed_deposit_flag": pl.Boolean,
            "insurance_flag": pl.Boolean,
            "mutual_fund_flag": pl.Boolean,
            "demat_account_flag": pl.Boolean,
            "wealth_management_flag": pl.Boolean,
            "products_count": pl.Int32,
        },
    )
