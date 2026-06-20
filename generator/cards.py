"""Card portfolio generator for the synthetic retail banking universe.

Generates Debit and Credit card records for customers based on their product holdings,
income levels, and accounts.
"""

from datetime import date
from typing import Optional
import numpy as np
import polars as pl

from config.constants import CREDIT_LIMIT_MULTIPLIER, CARD_ID_START, PRIMARY_ACCOUNT_TYPE
from config.personas import Persona
from config.simulation import SimulationConfig
from generator.spine import Spine


def generate_cards(
    spine: Spine,
    customer_df: pl.DataFrame,
    initial_products: pl.DataFrame,
    accounts_df: pl.DataFrame,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generates the static card_portfolio DataFrame.

    Args:
        spine: Spine containing customer IDs and personas.
        customer_df: Customer master table containing annual income.
        initial_products: Initial product holdings table.
        accounts_df: Account master table.
        config: Simulation configuration.
        rng: Optional seeded numpy random generator.

    Returns:
        pl.DataFrame: The card portfolio table.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    # Map customer_id to persona from spine
    spine_personas = {
        row["customer_id"]: row["persona"]
        for row in spine.simulation_state.to_dicts()
    }

    # Convert to dictionary mappings for faster lookups
    cust_data = {
        row["customer_id"]: {
            "annual_income": row["annual_income"],
            "persona": spine_personas[row["customer_id"]],
            "customer_since": row["customer_since"],
        }
        for row in customer_df.to_dicts()
    }

    prod_data = {
        row["customer_id"]: {
            "debit_card": row["debit_card_flag"],
            "credit_card": row["credit_card_flag"],
        }
        for row in initial_products.to_dicts()
    }

    # Group accounts by customer for easy debit card linking
    cust_accounts = {}
    for acc in accounts_df.to_dicts():
        cid = acc["customer_id"]
        if cid not in cust_accounts:
            cust_accounts[cid] = []
        cust_accounts[cid].append(acc)

    card_rows = []
    current_card_id = CARD_ID_START

    for cid in spine.simulation_state["customer_id"].to_list():
        c_info = cust_data[cid]
        p_info = prod_data[cid]
        c_accs = cust_accounts.get(cid, [])

        if not c_accs:
            # Cannot issue cards without accounts
            continue

        # 1. Debit Card Generation
        if p_info["debit_card"]:
            # Primary account selection logic: Savings first, otherwise Current
            primary_acc = None
            for acc in c_accs:
                if acc["account_type"] == PRIMARY_ACCOUNT_TYPE:
                    primary_acc = acc
                    break
            if primary_acc is None:
                primary_acc = c_accs[0]

            issue_date = primary_acc["open_date"]
            # Expiry date is issue_date + 5 years
            try:
                expiry_date = issue_date.replace(year=issue_date.year + 5)
            except ValueError:
                # Handle Feb 29 leap year cases
                expiry_date = date(issue_date.year + 5, 3, 1)

            network = rng.choice(["Visa", "Mastercard", "RuPay"], p=[0.40, 0.40, 0.20])

            card_rows.append(
                {
                    "card_id": current_card_id,
                    "customer_id": cid,
                    "card_type": "Debit",
                    "network": network,
                    "issue_date": issue_date,
                    "expiry_date": expiry_date,
                    "card_status": "Active",
                    "primary_card_flag": True,
                    "credit_limit": 0.0,
                    "rewards_program": "Standard Cashback",
                    "reward_tier": "Classic",
                }
            )
            current_card_id += 1

        # 2. Credit Card Generation
        if p_info["credit_card"]:
            issue_date = c_info["customer_since"]
            try:
                expiry_date = issue_date.replace(year=issue_date.year + 5)
            except ValueError:
                expiry_date = date(issue_date.year + 5, 3, 1)

            network = rng.choice(["Visa", "Mastercard", "RuPay"], p=[0.45, 0.35, 0.20])

            # Calculate credit limit floor and ceil based on income
            monthly_income = c_info["annual_income"] / 12.0
            raw_limit = monthly_income * CREDIT_LIMIT_MULTIPLIER
            # Round limit to the nearest 1000 INR, minimum of 10,000 INR
            credit_limit = float(max(10000.0, round(raw_limit, -3)))

            # Setup reward program and tiers based on persona
            p_val = c_info["persona"]
            if p_val == Persona.AFFLUENT_MULTI_PRODUCT.value:
                program = "Premium Travel"
                tier = "Platinum"
            elif p_val == Persona.DORMANT_WEALTHY.value:
                program = "Premium Rewards"
                tier = "Platinum"
            elif p_val == Persona.DIGITAL_NATIVE.value:
                program = "Online Shopper"
                tier = "Gold"
            elif p_val == Persona.SALARY_CORE.value:
                program = "Cashback Plus"
                tier = "Silver"
            elif p_val == Persona.COMPLAINT_PRONE_CHURNER.value:
                program = "Standard Cashback"
                tier = "Silver"
            else:
                program = "Basic Cashback"
                tier = "Classic"

            card_rows.append(
                {
                    "card_id": current_card_id,
                    "customer_id": cid,
                    "card_type": "Credit",
                    "network": network,
                    "issue_date": issue_date,
                    "expiry_date": expiry_date,
                    "card_status": "Active",
                    "primary_card_flag": True,
                    "credit_limit": credit_limit,
                    "rewards_program": program,
                    "reward_tier": tier,
                }
            )
            current_card_id += 1

    return pl.DataFrame(
        card_rows,
        schema={
            "card_id": pl.Int64,
            "customer_id": pl.Int64,
            "card_type": pl.String,
            "network": pl.String,
            "issue_date": pl.Date,
            "expiry_date": pl.Date,
            "card_status": pl.String,
            "primary_card_flag": pl.Boolean,
            "credit_limit": pl.Float64,
            "rewards_program": pl.String,
            "reward_tier": pl.String,
        },
    )
