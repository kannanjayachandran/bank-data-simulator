"""Loan master generator for the synthetic retail banking universe.

Generates Personal and Home loan records based on customer product holdings,
income levels, and open dates.
"""

from datetime import date
from typing import Optional
import numpy as np
import polars as pl

from config.constants import (
    HOME_LOAN_TENURE_RANGE,
    LOAN_ID_START,
    PERSONAL_LOAN_TENURE_RANGE,
    FOIR_LIMITS,
)
from config.simulation import SimulationConfig
from generator.branches import BRANCH_DATA
from generator.spine import Spine


# Map city to branch code for fast lookup
CITY_TO_BRANCH = {b["city"]: b["branch_code"] for b in BRANCH_DATA}


def compute_max_principal(max_emi: float, rate: float, tenure_months: int) -> float:
    """Computes the maximum principal that corresponds to a maximum monthly EMI."""
    R = rate / 12.0 / 100.0
    N = tenure_months
    if R == 0:
        return max_emi * N
    else:
        return max_emi * (((1 + R) ** N) - 1) / (R * ((1 + R) ** N))


def generate_loans(
    spine: Spine,
    customer_df: pl.DataFrame,
    initial_products: pl.DataFrame,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generates the static loan_master DataFrame.

    Args:
        spine: Spine containing customer IDs and personas.
        customer_df: Customer master table containing join dates, annual income, and cities.
        initial_products: Initial product holdings table.
        config: Simulation configuration.
        rng: Optional seeded numpy random generator.

    Returns:
        pl.DataFrame: The loan master table.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    # Convert to dictionaries for fast lookup
    cust_data = {
        row["customer_id"]: {
            "customer_since": row["customer_since"],
            "city": row["city"],
            "annual_income": row["annual_income"],
        }
        for row in customer_df.to_dicts()
    }

    prod_data = {
        row["customer_id"]: {
            "personal_loan": row["personal_loan_flag"],
            "home_loan": row["home_loan_flag"],
        }
        for row in initial_products.to_dicts()
    }

    loan_rows = []
    current_loan_id = LOAN_ID_START

    for cid in spine.simulation_state["customer_id"].to_list():
        c_info = cust_data[cid]
        p_info = prod_data[cid]

        branch_code = CITY_TO_BRANCH.get(c_info["city"], "B001")
        join_date = c_info["customer_since"]

        # Calculate max months between customer join date and simulation start
        # Limit to 1 to 24 months for loan disbursement relative to sim_start
        months_since_joined = (config.sim_start.year - join_date.year) * 12 + (config.sim_start.month - join_date.month)
        disbursement_months_ago = rng.integers(1, max(2, min(24, months_since_joined + 1)))

        # Subtract disbursement_months_ago from sim_start
        disb_year = config.sim_start.year - (disbursement_months_ago // 12)
        disb_month = config.sim_start.month - (disbursement_months_ago % 12)
        if disb_month <= 0:
            disb_year -= 1
            disb_month += 12
        disbursement_date = date(disb_year, disb_month, 1)

        # 1. Personal Loan Generation
        if p_info["personal_loan"]:
            interest_rate = float(rng.uniform(10.5, 15.0))
            # Sample personal loan tenure from [12, 24, 36, 48, 60] months
            tenure_months = int(rng.choice([12, 24, 36, 48, 60]))
            
            # Sanctioned amount based on annual income
            max_amt = max(50000.0, c_info["annual_income"] * 0.5)
            sanctioned_amount = float(max(50000.0, round(rng.uniform(50000.0, max_amt), -3)))

            # Apply FOIR Limit (Fix 1)
            monthly_income = c_info["annual_income"] / 12.0
            max_emi = monthly_income * FOIR_LIMITS["Personal Loan"]
            max_p = compute_max_principal(max_emi, interest_rate, tenure_months)
            sanctioned_amount = float(round(min(sanctioned_amount, max_p), -3))

            # EMI Calculation
            R = interest_rate / 12.0 / 100.0
            N = tenure_months
            P = sanctioned_amount
            if R == 0:
                emi = P / N
            else:
                emi = P * R * ((1 + R) ** N) / (((1 + R) ** N) - 1)
            emi_amount = float(round(emi, 2))

            # Maturity date
            mat_year = disbursement_date.year + (tenure_months // 12)
            mat_month = disbursement_date.month + (tenure_months % 12)
            if mat_month > 12:
                mat_year += 1
                mat_month -= 12
            maturity_date = date(mat_year, mat_month, 1)

            loan_rows.append(
                {
                    "loan_id": current_loan_id,
                    "customer_id": cid,
                    "branch_code": branch_code,
                    "loan_type": "Personal Loan",
                    "sanctioned_amount": sanctioned_amount,
                    "disbursement_date": disbursement_date,
                    "interest_rate": float(round(interest_rate, 3)),
                    "tenure_months": tenure_months,
                    "emi_amount": emi_amount,
                    "loan_purpose": rng.choice(["Medical", "Education", "Wedding", "Travel", "Home Renovation"]),
                    "origination_channel": rng.choice(["Branch", "Online", "Agent"], p=[0.40, 0.40, 0.20]),
                    "loan_status": "Active",
                    "maturity_date": maturity_date,
                }
            )
            current_loan_id += 1

        # 2. Home Loan Generation
        if p_info["home_loan"]:
            interest_rate = float(rng.uniform(8.2, 9.5))
            # Sample home loan tenure from [120, 180, 240] months
            tenure_months = int(rng.choice([120, 180, 240]))
            
            # Sanctioned amount based on annual income
            max_amt = max(1500000.0, c_info["annual_income"] * 5.0)
            sanctioned_amount = float(max(1500000.0, round(rng.uniform(1500000.0, max_amt), -4)))

            # Apply FOIR Limit (Fix 1)
            monthly_income = c_info["annual_income"] / 12.0
            max_emi = monthly_income * FOIR_LIMITS["Home Loan"]
            max_p = compute_max_principal(max_emi, interest_rate, tenure_months)
            sanctioned_amount = float(round(min(sanctioned_amount, max_p), -4))

            # EMI Calculation
            R = interest_rate / 12.0 / 100.0
            N = tenure_months
            P = sanctioned_amount
            if R == 0:
                emi = P / N
            else:
                emi = P * R * ((1 + R) ** N) / (((1 + R) ** N) - 1)
            emi_amount = float(round(emi, 2))

            # Maturity date
            mat_year = disbursement_date.year + (tenure_months // 12)
            mat_month = disbursement_date.month + (tenure_months % 12)
            if mat_month > 12:
                mat_year += 1
                mat_month -= 12
            maturity_date = date(mat_year, mat_month, 1)

            loan_rows.append(
                {
                    "loan_id": current_loan_id,
                    "customer_id": cid,
                    "branch_code": branch_code,
                    "loan_type": "Home Loan",
                    "sanctioned_amount": sanctioned_amount,
                    "disbursement_date": disbursement_date,
                    "interest_rate": float(round(interest_rate, 3)),
                    "tenure_months": tenure_months,
                    "emi_amount": emi_amount,
                    "loan_purpose": "Property Purchase",
                    "origination_channel": rng.choice(["Branch", "Online", "Agent"], p=[0.50, 0.30, 0.20]),
                    "loan_status": "Active",
                    "maturity_date": maturity_date,
                }
            )
            current_loan_id += 1

    return pl.DataFrame(
        loan_rows,
        schema={
            "loan_id": pl.Int64,
            "customer_id": pl.Int64,
            "branch_code": pl.String,
            "loan_type": pl.String,
            "sanctioned_amount": pl.Float64,
            "disbursement_date": pl.Date,
            "interest_rate": pl.Float64,
            "tenure_months": pl.Int32,
            "emi_amount": pl.Float64,
            "loan_purpose": pl.String,
            "origination_channel": pl.String,
            "loan_status": pl.String,
            "maturity_date": pl.Date,
        },
    )
