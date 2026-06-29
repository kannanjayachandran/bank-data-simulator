"""Customer monthly activity generator for the synthetic retail banking universe.

Generates the customer_monthly_activity DataFrame for a given snapshot month
based on persona configurations, transaction summaries, and active events.
"""

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from config.personas import Persona


def generate_monthly_activity(
    customer_ids: List[int],
    snapshot_month: date,
    personas: List[str],
    rng: np.random.Generator,
    txn_aggregates: Optional[Dict[int, dict]] = None,
    active_events_dict: Optional[Dict[int, List[str]]] = None,
) -> pl.DataFrame:
    """Generates the customer_monthly_activity DataFrame for a given month.

    Args:
        customer_ids: List of customer IDs to generate activity for.
        snapshot_month: Snapshot month (first day of month).
        personas: List of persona strings corresponding to customer_ids.
        rng: Seeded numpy random generator.
        txn_aggregates: Optional dictionary mapping customer_id to transaction aggregates of the month:
                        {"debit_count": int, "credit_count": int, "debit_amount": float,
                         "credit_amount": float, "unique_merchants": int, "cash_withdrawal_count": int,
                         "card_present_count": int, "card_not_present_count": int}
        active_events_dict: Optional dictionary mapping customer_id to list of active event names in the month.

    Returns:
        pl.DataFrame: The customer monthly activity table.
    """
    if txn_aggregates is None:
        txn_aggregates = {}
    if active_events_dict is None:
        active_events_dict = {}

    rows = []
    for cid, pers_name in zip(customer_ids, personas):
        p_enum = Persona(pers_name)
        events = active_events_dict.get(cid, [])
        txns = txn_aggregates.get(
            cid,
            {
                "debit_count": 0,
                "credit_count": 0,
                "debit_amount": 0.0,
                "credit_amount": 0.0,
                "unique_merchants": 0,
                "cash_withdrawal_count": 0,
                "card_present_count": 0,
                "card_not_present_count": 0,
            },
        )

        # 1. Logins Poisson parameters by persona
        if p_enum == Persona.DIGITAL_NATIVE:
            login_lambda = 30.0
        elif p_enum == Persona.AFFLUENT_MULTI_PRODUCT:
            login_lambda = 20.0
        elif p_enum == Persona.SALARY_CORE:
            login_lambda = 15.0
        elif p_enum in [Persona.CREDIT_STRESSED, Persona.COMPLAINT_PRONE_CHURNER]:
            login_lambda = 12.0
        else:  # Persona.DORMANT_WEALTHY
            login_lambda = 2.0

        # Event-driven modifiers on digital activity
        if "bank_service_failure" in events:
            # Login frequency drops in the month following a service failure
            login_lambda *= 0.5
        if "campaign_exposure" in events:
            login_lambda *= 1.3

        # Sample logins
        login_count = int(rng.poisson(max(0.5, login_lambda)))

        # Split logins into mobile app vs internet banking sessions
        if p_enum == Persona.DIGITAL_NATIVE:
            mobile_share = 0.90
        elif p_enum == Persona.DORMANT_WEALTHY:
            mobile_share = 0.30
        else:
            mobile_share = 0.70

        mobile_sessions = int(rng.binomial(login_count, mobile_share))
        internet_sessions = max(0, login_count - mobile_sessions)

        # 2. Branch visits and ATM transactions (based on preferences)
        if p_enum == Persona.DORMANT_WEALTHY:
            branch_lambda = 0.8
            atm_lambda = 2.0
        elif p_enum == Persona.SALARY_CORE:
            branch_lambda = 0.4
            atm_lambda = 3.0
        elif p_enum == Persona.DIGITAL_NATIVE:
            branch_lambda = 0.02
            atm_lambda = 0.5
        else:
            branch_lambda = 0.2
            atm_lambda = 1.5

        # Event-driven branch/ATM modifiers
        if "relocation" in events:
            branch_lambda += 1.0  # visiting branch to update details

        branch_visits = int(rng.poisson(branch_lambda))
        atm_transactions = int(rng.poisson(atm_lambda))

        # 3. Transaction summary columns (use aggregates if present, otherwise mock)
        debit_txn_count = txns["debit_count"]
        credit_txn_count = txns["credit_count"]
        total_debit_amount = round(float(txns["debit_amount"]), 2)
        total_credit_amount = round(float(txns["credit_amount"]), 2)

        if debit_txn_count > 0:
            avg_txn_val = round(total_debit_amount / debit_txn_count, 2)
        else:
            avg_txn_val = 0.0

        unique_merchants = txns["unique_merchants"]
        cash_withdrawal_count = txns["cash_withdrawal_count"]
        card_present_txn_count = txns["card_present_count"]
        card_not_present_txn_count = txns["card_not_present_count"]

        # Days since last event
        days_since_last_login = (
            int(rng.integers(0, 30))
            if login_count == 0
            else int(rng.choice([0, 1, 2, 3, 4]))
        )
        days_since_last_txn = (
            int(rng.integers(0, 30))
            if debit_txn_count == 0
            else int(rng.choice([0, 1, 2, 3, 4, 5]))
        )

        rows.append(
            {
                "customer_id": cid,
                "snapshot_month": snapshot_month,
                "login_count": login_count,
                "mobile_app_sessions": mobile_sessions,
                "internet_banking_sessions": internet_sessions,
                "atm_transactions": atm_transactions,
                "branch_visits": branch_visits,
                "debit_txn_count": debit_txn_count,
                "credit_txn_count": credit_txn_count,
                "total_debit_amount": total_debit_amount,
                "total_credit_amount": total_credit_amount,
                "avg_transaction_value": avg_txn_val,
                "unique_merchants": unique_merchants,
                "cash_withdrawal_count": cash_withdrawal_count,
                "card_present_txn_count": card_present_txn_count,
                "card_not_present_txn_count": card_not_present_txn_count,
                "days_since_last_txn": days_since_last_txn,
                "days_since_last_login": days_since_last_login,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "customer_id": pl.Int64,
            "snapshot_month": pl.Date,
            "login_count": pl.Int32,
            "mobile_app_sessions": pl.Int32,
            "internet_banking_sessions": pl.Int32,
            "atm_transactions": pl.Int32,
            "branch_visits": pl.Int32,
            "debit_txn_count": pl.Int32,
            "credit_txn_count": pl.Int32,
            "total_debit_amount": pl.Float64,
            "total_credit_amount": pl.Float64,
            "avg_transaction_value": pl.Float64,
            "unique_merchants": pl.Int32,
            "cash_withdrawal_count": pl.Int32,
            "card_present_txn_count": pl.Int32,
            "card_not_present_txn_count": pl.Int32,
            "days_since_last_txn": pl.Int32,
            "days_since_last_login": pl.Int32,
        },
    )
