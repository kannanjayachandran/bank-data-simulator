"""Account master generator for the synthetic retail banking universe.

Generates Savings and Current account master records based on initial product holdings
and customer locations.
"""

from typing import Optional
import numpy as np
import polars as pl

from config.constants import ACCOUNT_ID_START
from config.personas import Persona
from config.simulation import SimulationConfig
from generator.branches import BRANCH_DATA
from generator.spine import Spine


# Map city to branch code for fast lookup
CITY_TO_BRANCH = {b["city"]: b["branch_code"] for b in BRANCH_DATA}


def generate_accounts(
    spine: Spine,
    customer_df: pl.DataFrame,
    initial_products: pl.DataFrame,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generates the static account_master DataFrame.

    Args:
        spine: Spine containing customer IDs and personas.
        customer_df: Customer master table containing join dates and cities.
        initial_products: Initial product holdings table.
        config: Simulation configuration.
        rng: Optional seeded numpy random generator.

    Returns:
        pl.DataFrame: The account master table.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    # Map customer_id to persona from spine
    spine_personas = {
        row["customer_id"]: row["persona"]
        for row in spine.simulation_state.to_dicts()
    }

    # Convert customer and product data to dicts for fast iteration
    cust_data = {
        row["customer_id"]: {
            "customer_since": row["customer_since"],
            "city": row["city"],
            "persona": spine_personas[row["customer_id"]],
            "annual_income": row["annual_income"],
        }
        for row in customer_df.to_dicts()
    }

    prod_data = {
        row["customer_id"]: {
            "savings": row["savings_flag"],
            "current": row["current_flag"],
        }
        for row in initial_products.to_dicts()
    }

    account_rows = []
    current_account_id = ACCOUNT_ID_START

    for cid in spine.simulation_state["customer_id"].to_list():
        c_info = cust_data[cid]
        p_info = prod_data[cid]

        # Home branch mapping based on city
        branch_code = CITY_TO_BRANCH.get(c_info["city"], "B001")

        # 1. Savings account creation
        if p_info["savings"]:
            # Salary flag is True for salary_core and complaint_prone_churner personas
            salary_flag = c_info["persona"] in [
                Persona.SALARY_CORE.value,
                Persona.COMPLAINT_PRONE_CHURNER.value,
            ]

            account_rows.append(
                {
                    "account_id": current_account_id,
                    "customer_id": cid,
                    "branch_code": branch_code,
                    "account_type": "Savings",
                    "open_date": c_info["customer_since"],
                    "account_status": "Active",
                    "account_currency": "INR",
                    "salary_account_flag": salary_flag,
                    "overdraft_limit": 0.0,
                    "account_close_date": None,
                }
            )
            current_account_id += 1

        # 2. Current account creation
        if p_info["current"]:
            # Set overdraft limit based on persona income level
            if c_info["persona"] == Persona.AFFLUENT_MULTI_PRODUCT.value:
                overdraft = float(rng.choice([50000.0, 100000.0], p=[0.40, 0.60]))
            elif c_info["persona"] == Persona.DORMANT_WEALTHY.value:
                overdraft = float(rng.choice([50000.0, 100000.0], p=[0.50, 0.50]))
            else:
                overdraft = float(rng.choice([25000.0, 50000.0], p=[0.70, 0.30]))

            account_rows.append(
                {
                    "account_id": current_account_id,
                    "customer_id": cid,
                    "branch_code": branch_code,
                    "account_type": "Current",
                    "open_date": c_info["customer_since"],
                    "account_status": "Active",
                    "account_currency": "INR",
                    "salary_account_flag": False,
                    "overdraft_limit": overdraft,
                    "account_close_date": None,
                }
            )
            current_account_id += 1

    return pl.DataFrame(
        account_rows,
        schema={
            "account_id": pl.Int64,
            "customer_id": pl.Int64,
            "branch_code": pl.String,
            "account_type": pl.String,
            "open_date": pl.Date,
            "account_status": pl.String,
            "account_currency": pl.String,
            "salary_account_flag": pl.Boolean,
            "overdraft_limit": pl.Float64,
            "account_close_date": pl.Date,
        },
    )
