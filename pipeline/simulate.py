"""Main simulation engine for Churn Compass retail banking data generator.

Orchestrates the monthly loop, dynamic event evaluation, transaction generation,
balance/card/loan tracking, product holdings evolution, churn scoring, and data partition writing.
"""

from datetime import date, datetime, time, timedelta
from typing import Dict, List, Any, Optional, Set
import numpy as np
import polars as pl
import os

from config.simulation import SimulationConfig
from config.personas import Persona, PERSONA_CONFIGS
from config.events import HiddenEvent, CONDITIONAL_EVENT_BASELINES
from config.constants import (
    ACCOUNT_ID_START,
    CARD_ID_START,
    LOAN_ID_START,
    TRANSACTION_ID_START,
    COMPLAINT_ID_START,
    FEEDBACK_ID_START,
    PRIMARY_ACCOUNT_TYPE,
    CREDIT_LIMIT_MULTIPLIER,
)
from generator.branches import generate_branches
from generator.customers import generate_customers
from generator.products import generate_initial_products
from generator.accounts import generate_accounts
from generator.cards import generate_cards
from generator.loans import generate_loans
from generator.spine import generate_spine
from generator.transactions import (
    generate_salary_credit,
    generate_non_salary_income,
    generate_regular_transactions,
    generate_fee_or_charge,
    generate_monthly_regular_transactions,
    generate_monthly_non_salary_income,
)
from generator.activity import generate_monthly_activity
from generator.digital import generate_monthly_digital
from generator.churn import calculate_churn, ChurnInput, ChurnResult
from generator.complaints import generate_complaints_for_month, resolve_complaints_for_month
from generator.feedback import generate_feedback_for_month
from generator.labels import generate_churn_labels, generate_feature_snapshots


def add_months(d: date, m: int) -> date:
    """Helper to add months to a date."""
    y_offset = (d.month + m - 1) // 12
    m_offset = (d.month + m - 1) % 12 + 1
    return date(d.year + y_offset, m_offset, 1)


def get_days_in_month(d: date) -> int:
    """Helper to get number of days in a month."""
    if d.month == 12:
        next_month = date(d.year + 1, 1, 1)
    else:
        next_month = date(d.year, d.month + 1, 1)
    return (next_month - d).days


def run_simulation(
    config: SimulationConfig,
    streaming: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, pl.DataFrame]:
    """Runs the full simulation pipeline with O(1) indexed lookups for speed.

    Args:
        config: Simulation configuration.
        streaming: If True, writes monthly snapshots/transactions incrementally to Parquet and clears memory.
        output_dir: Output directory path for streaming mode.

    Returns:
        Dict[str, pl.DataFrame]: Dictionary containing the resulting tables.
    """
    rng = np.random.default_rng(config.seed)

    # 1. Generate Static Master Tables
    branches_df = generate_branches()
    spine = generate_spine(config)
    initial_products = generate_initial_products(spine, config, rng)
    customer_df = generate_customers(spine, config, rng)
    accounts_df = generate_accounts(spine, customer_df, initial_products, config, rng)
    cards_df = generate_cards(spine, customer_df, initial_products, accounts_df, config, rng)
    loans_df = generate_loans(spine, customer_df, initial_products, config, rng)

    # 2. Convert DataFrames to dicts/lists for runtime modifications
    customers = customer_df.to_dicts()
    customer_personas = {
        row["customer_id"]: row["persona"]
        for row in spine.simulation_state.to_dicts()
    }
    for c in customers:
        c["persona"] = customer_personas[c["customer_id"]]

    accounts = accounts_df.to_dicts()
    cards = cards_df.to_dicts()
    loans = loans_df.to_dicts()

    # Index maps for O(1) lookups
    accounts_by_customer = {}
    for acc in accounts:
        accounts_by_customer.setdefault(acc["customer_id"], []).append(acc)

    cards_by_customer = {}
    for card in cards:
        cards_by_customer.setdefault(card["customer_id"], []).append(card)

    loans_by_customer = {}
    for ln in loans:
        loans_by_customer.setdefault(ln["customer_id"], []).append(ln)

    complaints_by_customer = {}
    
    # Track product holdings dynamically
    # customer_id -> dict of product flag names to bool
    product_holdings = {
        row["customer_id"]: {
            "savings_account_flag": row["savings_flag"],
            "current_account_flag": row["current_flag"],
            "credit_card_flag": row["credit_card_flag"],
            "personal_loan_flag": row["personal_loan_flag"],
            "home_loan_flag": row["home_loan_flag"],
            "fixed_deposit_flag": row["fixed_deposit_flag"],
            "insurance_flag": row["insurance_flag"],
            "mutual_fund_flag": row["mutual_fund_flag"],
            "demat_account_flag": row["demat_account_flag"],
            "wealth_management_flag": row["wealth_management_flag"],
        }
        for row in initial_products.to_dicts()
    }

    # Initialize State Variables
    running_balances = {}  # account_id -> float
    for acc in accounts:
        cid = acc["customer_id"]
        c_p = spine_state_persona = customer_personas[cid]
        # Initialize running balance based on persona
        if c_p == Persona.AFFLUENT_MULTI_PRODUCT.value:
            bal = rng.uniform(500000.0, 1500000.0)
        elif c_p == Persona.DORMANT_WEALTHY.value:
            bal = rng.uniform(1500000.0, 5000000.0)
        elif c_p == Persona.SALARY_CORE.value:
            bal = rng.uniform(30000.0, 150000.0)
        elif c_p == Persona.DIGITAL_NATIVE.value:
            bal = rng.uniform(20000.0, 80000.0)
        elif c_p == Persona.CREDIT_STRESSED.value:
            bal = rng.uniform(5000.0, 25000.0)
        else:
            bal = rng.uniform(10000.0, 50000.0)
        running_balances[acc["account_id"]] = float(round(bal, 2))

    running_card_spends = {c["card_id"]: 0.0 for c in cards if c["card_type"] == "Credit"}

    # Initialize loan states
    running_loans = {}  # loan_id -> outstanding_balance, dpd_days, status
    for ln in loans:
        disb = ln["disbursement_date"]
        months_diff = (config.sim_start.year - disb.year) * 12 + (config.sim_start.month - disb.month)
        outstanding = ln["sanctioned_amount"]
        r = ln["interest_rate"] / 12.0 / 100.0
        emi = ln["emi_amount"]
        for _ in range(max(0, months_diff)):
            interest = outstanding * r
            principal = emi - interest
            outstanding = max(0.0, outstanding - principal)
        
        running_loans[ln["loan_id"]] = {
            "outstanding_balance": outstanding,
            "dpd_days": 0,
            "status": "Active" if outstanding > 0 else "Closed",
        }

    # Tracking metrics per customer
    months_without_salary = {c["customer_id"]: 0 for c in customers}
    service_failures_2m = {c["customer_id"]: 0 for c in customers}
    digital_inactive_months = {c["customer_id"]: 0 for c in customers}
    recent_service_failures = {c["customer_id"]: False for c in customers}
    products_count_drop = {c["customer_id"]: False for c in customers}
    previous_products_count = {c["customer_id"]: sum(product_holdings[c["customer_id"]].values()) for c in customers}

    # Dynamic ID offsets
    next_txn_id = TRANSACTION_ID_START
    next_complaint_id = COMPLAINT_ID_START
    next_feedback_id = FEEDBACK_ID_START
    next_account_id = ACCOUNT_ID_START + config.n_customers * 2
    next_card_id = CARD_ID_START + config.n_customers * 2
    next_loan_id = LOAN_ID_START + config.n_customers * 2

    # Global lists to accumulate snapshots
    all_transactions = []
    account_snapshots = []
    card_snapshots = []
    loan_snapshots = []
    product_holdings_snapshots = []
    activity_snapshots = []
    digital_snapshots = []
    all_complaints = []
    all_feedback = []

    # Map spine state for easy runtime updates
    spine_state = {
        row["customer_id"]: {
            "persona": row["persona"],
            "low_sensitivity_segment": row["low_sensitivity_segment"],
            "churn_month": None,
            "churned_flag": False,
            "churn_reason": None,
            "active_months_generated": 0,
        }
        for row in spine.simulation_state.to_dicts()
    }

    # Cache pre-scheduled events for fast lookup
    pre_scheduled_events_cache = {}
    for evt in spine.scheduled_events.to_dicts():
        cid = evt["customer_id"]
        m_date = evt["event_month"]
        pre_scheduled_events_cache.setdefault(cid, {}).setdefault(m_date, []).append(evt["event_type"])

    # Track dynamically fired events
    dynamic_events_fired = []

    # Monthly Loop
    for month_idx in range(config.sim_months):
        snapshot_month = add_months(config.sim_start, month_idx)
        days_in_month = get_days_in_month(snapshot_month)

        # 1. Filter out already churned customers
        active_customers = [
            c for c in customers
            if not spine_state[c["customer_id"]]["churned_flag"]
        ]
        active_cids = {c["customer_id"] for c in active_customers}

        if not active_customers:
            break

        customer_events: Dict[int, Set[str]] = {cid: set() for cid in active_cids}

        # Step 2: Evaluate and fire events
        for c in active_customers:
            cid = c["customer_id"]
            persona = Persona(c["persona"])

            # Evaluate Pre-scheduled Unconditional Events
            p_sched = pre_scheduled_events_cache.get(cid, {}).get(snapshot_month, [])
            for evt_name in p_sched:
                customer_events[cid].add(evt_name)

            # Evaluate Conditional Events (O(1) lookup on cards and loans)
            cust_cards = [card for card in cards_by_customer.get(cid, []) if card["card_status"] == "Active"]
            if cust_cards:
                prob = CONDITIONAL_EVENT_BASELINES[persona][HiddenEvent.CARD_DECLINE_SPIKE]
                if rng.random() < prob:
                    customer_events[cid].add(HiddenEvent.CARD_DECLINE_SPIKE.value)
                    dynamic_events_fired.append({
                        "customer_id": cid,
                        "event_month": snapshot_month,
                        "event_type": HiddenEvent.CARD_DECLINE_SPIKE.value,
                        "is_fired": True
                    })

            digitally_active = (persona == Persona.DIGITAL_NATIVE) or (rng.random() < 0.6)
            if digitally_active:
                prob = CONDITIONAL_EVENT_BASELINES[persona][HiddenEvent.CAMPAIGN_EXPOSURE]
                if rng.random() < prob:
                    customer_events[cid].add(HiddenEvent.CAMPAIGN_EXPOSURE.value)
                    dynamic_events_fired.append({
                        "customer_id": cid,
                        "event_month": snapshot_month,
                        "event_type": HiddenEvent.CAMPAIGN_EXPOSURE.value,
                        "is_fired": True
                    })

                    # Evaluate campaign success -> product adoption
                    success_rates = {
                        Persona.SALARY_CORE: 0.20,
                        Persona.AFFLUENT_MULTI_PRODUCT: 0.30,
                        Persona.DIGITAL_NATIVE: 0.25,
                        Persona.CREDIT_STRESSED: 0.15,
                        Persona.DORMANT_WEALTHY: 0.10,
                        Persona.COMPLAINT_PRONE_CHURNER: 0.10,
                    }
                    if rng.random() < success_rates[persona]:
                        holdings = product_holdings[cid]
                        eligible_products = [
                            k for k, v in holdings.items()
                            if not v and k != "savings_account_flag"
                        ]
                        if eligible_products:
                            new_prod = rng.choice(eligible_products)
                            holdings[new_prod] = True

                            # Append and Index new entities dynamically
                            if new_prod == "current_account_flag":
                                branch_code = next(b["branch_code"] for b in branches_df.to_dicts() if b["city"] == c["city"])
                                new_acc = {
                                    "account_id": next_account_id,
                                    "customer_id": cid,
                                    "branch_code": branch_code,
                                    "account_type": "Current",
                                    "open_date": snapshot_month,
                                    "account_status": "Active",
                                    "account_currency": "INR",
                                    "salary_account_flag": False,
                                    "overdraft_limit": float(rng.choice([25000.0, 50000.0])),
                                    "account_close_date": None,
                                }
                                accounts.append(new_acc)
                                accounts_by_customer.setdefault(cid, []).append(new_acc)
                                running_balances[next_account_id] = float(rng.uniform(10000.0, 30000.0))
                                next_account_id += 1

                            elif new_prod == "credit_card_flag":
                                limit = float(max(10000.0, round((c["annual_income"] / 12.0) * CREDIT_LIMIT_MULTIPLIER, -3)))
                                new_card = {
                                    "card_id": next_card_id,
                                    "customer_id": cid,
                                    "card_type": "Credit",
                                    "network": rng.choice(["Visa", "Mastercard"]),
                                    "issue_date": snapshot_month,
                                    "expiry_date": snapshot_month.replace(year=snapshot_month.year + 5),
                                    "card_status": "Active",
                                    "primary_card_flag": True,
                                    "credit_limit": limit,
                                    "rewards_program": "Standard Cashback",
                                    "reward_tier": "Silver",
                                }
                                cards.append(new_card)
                                cards_by_customer.setdefault(cid, []).append(new_card)
                                running_card_spends[next_card_id] = 0.0
                                next_card_id += 1

                            elif new_prod in ["personal_loan_flag", "home_loan_flag"]:
                                l_type = "Personal Loan" if new_prod == "personal_loan_flag" else "Home Loan"
                                sanctioned = 100000.0 if l_type == "Personal Loan" else 2000000.0
                                rate = 12.5 if l_type == "Personal Loan" else 8.5
                                tenure = 36 if l_type == "Personal Loan" else 180
                                
                                R = rate / 12.0 / 100.0
                                N = tenure
                                P = sanctioned
                                emi = P * R * ((1 + R) ** N) / (((1 + R) ** N) - 1)
                                emi_amount = float(round(emi, 2))

                                branch_code = next(b["branch_code"] for b in branches_df.to_dicts() if b["city"] == c["city"])
                                new_loan = {
                                    "loan_id": next_loan_id,
                                    "customer_id": cid,
                                    "branch_code": branch_code,
                                    "loan_type": l_type,
                                    "sanctioned_amount": sanctioned,
                                    "disbursement_date": snapshot_month,
                                    "interest_rate": rate,
                                    "tenure_months": tenure,
                                    "emi_amount": emi_amount,
                                    "loan_purpose": "General",
                                    "origination_channel": "Online",
                                    "loan_status": "Active",
                                    "maturity_date": snapshot_month + timedelta(days=tenure * 30),
                                }
                                loans.append(new_loan)
                                loans_by_customer.setdefault(cid, []).append(new_loan)
                                running_loans[next_loan_id] = {
                                    "outstanding_balance": sanctioned,
                                    "dpd_days": 0,
                                    "status": "Active",
                                }
                                next_loan_id += 1

            cust_loans = [l for l in loans_by_customer.get(cid, []) if running_loans[l["loan_id"]]["status"] == "Active"]
            if cust_loans:
                prob = CONDITIONAL_EVENT_BASELINES[persona][HiddenEvent.LOAN_DELINQUENCY_START]
                if rng.random() < prob:
                    customer_events[cid].add(HiddenEvent.LOAN_DELINQUENCY_START.value)
                    dynamic_events_fired.append({
                        "customer_id": cid,
                        "event_month": snapshot_month,
                        "event_type": HiddenEvent.LOAN_DELINQUENCY_START.value,
                        "is_fired": True
                    })

            cust_open_comps = [comp for comp in complaints_by_customer.get(cid, []) if comp["status"] == "Open"]
            if cust_open_comps:
                prob = CONDITIONAL_EVENT_BASELINES[persona][HiddenEvent.COMPLAINT_RESOLVED]
                if rng.random() < prob:
                    customer_events[cid].add(HiddenEvent.COMPLAINT_RESOLVED.value)
                    dynamic_events_fired.append({
                        "customer_id": cid,
                        "event_month": snapshot_month,
                        "event_type": HiddenEvent.COMPLAINT_RESOLVED.value,
                        "is_fired": True
                    })

        # Step 3: Generate Monthly Transactions & Snapshot Metrics Customer-by-Customer
        monthly_txns = []
        month_aggregates = {}

        resolved_complaints_this_month = set()

        # Pre-identify primary account and salary/non-salary accounts for all active customers
        non_salary_customers = []
        primary_accs = {}
        for c in active_customers:
            cid = c["customer_id"]
            c_accs = [a for a in accounts_by_customer.get(cid, []) if a["account_status"] == "Active"]
            primary_acc = None
            for acc in c_accs:
                if acc["account_type"] == PRIMARY_ACCOUNT_TYPE:
                    primary_acc = acc
                    break
            if primary_acc is None and c_accs:
                primary_acc = c_accs[0]
            
            if primary_acc:
                primary_accs[cid] = primary_acc
                if not primary_acc["salary_account_flag"]:
                    c_copy = dict(c)
                    c_copy["account_id"] = primary_acc["account_id"]
                    non_salary_customers.append(c_copy)

        # Batch generate non-salary income credits
        pregen_credits = generate_monthly_non_salary_income(
            non_salary_customers, customer_events, snapshot_month, next_txn_id, rng
        )
        next_txn_id += len(pregen_credits)

        # Batch generate regular debit transactions
        pregen_debits = generate_monthly_regular_transactions(
            active_customers, customer_events, snapshot_month, next_txn_id, rng
        )
        next_txn_id += len(pregen_debits)

        # Group pre-generated transactions by customer_id
        from collections import defaultdict
        pregen_credits_by_cust = defaultdict(list)
        for t in pregen_credits:
            pregen_credits_by_cust[t["customer_id"]].append(t)

        pregen_debits_by_cust = defaultdict(list)
        for t in pregen_debits:
            pregen_debits_by_cust[t["customer_id"]].append(t)

        for c in active_customers:
            cid = c["customer_id"]
            persona = Persona(c["persona"])
            events = customer_events[cid]
            c_accs = [a for a in accounts_by_customer.get(cid, []) if a["account_status"] == "Active"]

            txn_cnt = 0
            cred_cnt = 0
            debit_cnt = 0
            total_deb = 0.0
            total_cred = 0.0
            unique_merch = set()
            cash_with = 0
            card_pres = 0
            card_npres = 0

            primary_acc = primary_accs.get(cid)
            if not primary_acc:
                continue

            acc_id = primary_acc["account_id"]
            start_bal = running_balances[acc_id]
            cust_txns = []

            # 3.1 Salary Credit (Day 1)
            is_salary_delayed = "salary_delay" in events
            is_job_change = "salary_job_change" in events

            if primary_acc["salary_account_flag"]:
                if is_salary_delayed:
                    months_without_salary[cid] += 1
                elif is_job_change:
                    if rng.random() < 0.5:
                        months_without_salary[cid] += 1
                    else:
                        salary_amt = (c["annual_income"] / 12.0) * rng.uniform(0.9, 1.2)
                        cust_txns.append(
                            generate_salary_credit(next_txn_id, cid, acc_id, salary_amt, date(snapshot_month.year, snapshot_month.month, 1), c["city"], c["state"])
                        )
                        next_txn_id += 1
                        months_without_salary[cid] = 0
                else:
                    salary_amt = c["annual_income"] / 12.0
                    cust_txns.append(
                        generate_salary_credit(next_txn_id, cid, acc_id, salary_amt, date(snapshot_month.year, snapshot_month.month, 1), c["city"], c["state"])
                    )
                    next_txn_id += 1
                    months_without_salary[cid] = 0
            else:
                cust_txns.extend(pregen_credits_by_cust.get(cid, []))

            # 3.2 Regular Transactions
            cc = [card for card in cards_by_customer.get(cid, []) if card["card_type"] == "Credit" and card["card_status"] == "Active"]
            for t in pregen_debits_by_cust.get(cid, []):
                t["account_id"] = acc_id
                if cc and rng.random() < 0.40:
                    card_id = cc[0]["card_id"]
                    running_card_spends[card_id] += t["amount"]
                else:
                    cust_txns.append(t)

            # 3.3 Loan EMI Payment (Day 10)
            cust_loans = loans_by_customer.get(cid, [])
            for ln in cust_loans:
                loan_id = ln["loan_id"]
                ln_state = running_loans[loan_id]

                if ln_state["status"] == "Active":
                    emi = ln["emi_amount"]
                    interest_rate = ln["interest_rate"]
                    outstanding = ln_state["outstanding_balance"]
                    
                    interest = outstanding * (interest_rate / 12.0 / 100.0)
                    principal = emi - interest
                    principal = min(principal, outstanding)
                    actual_emi = principal + interest

                    current_temp_bal = start_bal + sum(t["amount"] if t["direction"] == "Credit" else -t["amount"] for t in cust_txns)
                    overdraft = primary_acc["overdraft_limit"]
                    
                    is_delinquent_now = "loan_delinquency_start" in events or (current_temp_bal + overdraft < actual_emi)

                    if not is_delinquent_now:
                        cust_txns.append({
                            "transaction_id": next_txn_id,
                            "account_id": acc_id,
                            "customer_id": cid,
                            "txn_timestamp": datetime.combine(date(snapshot_month.year, snapshot_month.month, 10), time(10, 0, 0)),
                            "txn_date": date(snapshot_month.year, snapshot_month.month, 10),
                            "txn_month": snapshot_month,
                            "txn_type": "Loan EMI Payment",
                            "direction": "Debit",
                            "channel": "System",
                            "amount": round(actual_emi, 2),
                            "currency": "INR",
                            "merchant_category": "Financial Services",
                            "merchant_name": "Bank Loan Dept",
                            "counterparty_type": "Bank",
                            "city": c["city"],
                            "state": c["state"],
                            "is_salary_credit": False,
                            "is_fee": False,
                            "is_reversal": False,
                            "balance_after_txn": 0.0,
                        })
                        next_txn_id += 1
                        
                        ln_state["outstanding_balance"] = max(0.0, outstanding - principal)
                        ln_state["dpd_days"] = 0
                        if ln_state["outstanding_balance"] <= 0.0:
                            ln_state["status"] = "Closed"
                    else:
                        ln_state["dpd_days"] += 30
                        if ln_state["dpd_days"] >= 90:
                            ln_state["dpd_days"] = 90
                            ln_state["status"] = "Delinquent"

            # 3.4 Credit Card Payment (Day 25)
            cust_cc = [card for card in cards_by_customer.get(cid, []) if card["card_type"] == "Credit"]
            for cc in cust_cc:
                card_id = cc["card_id"]
                spend = running_card_spends[card_id]
                if spend > 0:
                    current_temp_bal = start_bal + sum(t["amount"] if t["direction"] == "Credit" else -t["amount"] for t in cust_txns)
                    overdraft = primary_acc["overdraft_limit"]
                    
                    if current_temp_bal + overdraft >= spend:
                        cust_txns.append({
                            "transaction_id": next_txn_id,
                            "account_id": acc_id,
                            "customer_id": cid,
                            "txn_timestamp": datetime.combine(date(snapshot_month.year, snapshot_month.month, 25), time(18, 0, 0)),
                            "txn_date": date(snapshot_month.year, snapshot_month.month, 25),
                            "txn_month": snapshot_month,
                            "txn_type": "Credit Card Payment",
                            "direction": "Debit",
                            "channel": "System",
                            "amount": round(spend, 2),
                            "currency": "INR",
                            "merchant_category": "Credit Card",
                            "merchant_name": "Bank Card Division",
                            "counterparty_type": "Bank",
                            "city": c["city"],
                            "state": c["state"],
                            "is_salary_credit": False,
                            "is_fee": False,
                            "is_reversal": False,
                            "balance_after_txn": 0.0,
                        })
                        next_txn_id += 1
                        running_card_spends[card_id] = 0.0
                    else:
                        min_due = spend * 0.05
                        if current_temp_bal + overdraft >= min_due:
                            cust_txns.append({
                                "transaction_id": next_txn_id,
                                "account_id": acc_id,
                                "customer_id": cid,
                                "txn_timestamp": datetime.combine(date(snapshot_month.year, snapshot_month.month, 25), time(18, 0, 0)),
                                "txn_date": date(snapshot_month.year, snapshot_month.month, 25),
                                "txn_month": snapshot_month,
                                "txn_type": "Credit Card Payment",
                                "direction": "Debit",
                                "channel": "System",
                                "amount": round(min_due, 2),
                                "currency": "INR",
                                "merchant_category": "Credit Card",
                                "merchant_name": "Bank Card Division",
                                "counterparty_type": "Bank",
                                "city": c["city"],
                                "state": c["state"],
                                "is_salary_credit": False,
                                "is_fee": False,
                                "is_reversal": False,
                                "balance_after_txn": 0.0,
                            })
                            next_txn_id += 1
                            running_card_spends[card_id] = (spend - min_due) * 1.02
                        else:
                            running_card_spends[card_id] = spend * 1.03

            # 3.5 System Fees (Day 28)
            is_fee_hike = "fee_hike_or_service_charge" in events
            if is_fee_hike or rng.random() < 0.10:
                fee_amt = rng.uniform(100.0, 500.0) if is_fee_hike else rng.uniform(50.0, 150.0)
                cust_txns.append(
                    generate_fee_or_charge(next_txn_id, cid, acc_id, date(snapshot_month.year, snapshot_month.month, 28), fee_amt, c["city"], c["state"])
                )
                next_txn_id += 1

            cust_txns.sort(key=lambda t: t["txn_timestamp"])
            
            running_bal = start_bal
            txn_balances = []
            
            for t in cust_txns:
                if t["direction"] == "Credit":
                    running_bal += t["amount"]
                else:
                    running_bal -= t["amount"]
                t["balance_after_txn"] = round(running_bal, 2)
                txn_balances.append(running_bal)
                
                txn_cnt += 1
                if t["direction"] == "Credit":
                    cred_cnt += 1
                    total_cred += t["amount"]
                else:
                    debit_cnt += 1
                    total_deb += t["amount"]
                unique_merch.add(t["merchant_name"])
                if t["channel"] == "ATM":
                    cash_with += 1
                if t["channel"] == "POS":
                    card_pres += 1
                elif t["channel"] in ["UPI", "Internet Banking", "Mobile App"]:
                    card_npres += 1

            running_balances[acc_id] = float(round(running_bal, 2))

            month_aggregates[cid] = {
                "debit_count": debit_cnt,
                "credit_count": cred_cnt,
                "debit_amount": total_deb,
                "credit_amount": total_cred,
                "unique_merchants": len(unique_merch),
                "cash_withdrawal_count": cash_with,
                "card_present_count": card_pres,
                "card_not_present_count": card_npres,
            }

            monthly_txns.extend(cust_txns)

            current_bal = float(round(running_bal, 2))
            avg_bal = np.mean(txn_balances) if txn_balances else current_bal
            min_bal = np.min(txn_balances) if txn_balances else current_bal
            max_bal = np.max(txn_balances) if txn_balances else current_bal

            account_snapshots.append({
                "account_id": acc_id,
                "snapshot_month": snapshot_month,
                "current_balance": current_bal,
                "average_monthly_balance": float(round(avg_bal, 2)),
                "min_balance_30d": float(round(min_bal, 2)),
                "max_balance_30d": float(round(max_bal, 2)),
                "deposit_count": cred_cnt,
                "withdrawal_count": debit_cnt,
                "debit_txn_count": debit_cnt,
                "credit_txn_count": cred_cnt,
                "fee_charged_amount": float(round(sum(t["amount"] for t in cust_txns if t["is_fee"]), 2)),
                "salary_credit_amount": float(round(sum(t["amount"] for t in cust_txns if t["is_salary_credit"]), 2)),
                "account_status": primary_acc["account_status"],
            })

            cust_cc = cards_by_customer.get(cid, [])
            for cc in cust_cc:
                if cc["card_type"] == "Credit":
                    spend = running_card_spends[cc["card_id"]]
                    limit = cc["credit_limit"]
                    util = spend / limit if limit > 0 else 0.0
                    card_snapshots.append({
                        "card_id": cc["card_id"],
                        "snapshot_month": snapshot_month,
                        "monthly_spend_amount": float(round(spend, 2)),
                        "monthly_txn_count": int(rng.poisson(4.0)) if spend > 0 else 0,
                        "cash_advance_amount": 0.0,
                        "utilization_rate": float(round(util, 4)),
                        "min_due_amount": float(round(spend * 0.05, 2)),
                        "payment_made_amount": float(round(spend, 2)) if spend > 0 else 0.0,
                        "rewards_points_balance": int(rng.integers(100, 1000)) if spend > 0 else 0,
                        "card_status": cc["card_status"],
                        "delinquency_flag": bool(spend > limit),
                    })
                else:
                    card_snapshots.append({
                        "card_id": cc["card_id"],
                        "snapshot_month": snapshot_month,
                        "monthly_spend_amount": float(round(total_deb * 0.3, 2)),
                        "monthly_txn_count": int(debit_cnt),
                        "cash_advance_amount": 0.0,
                        "utilization_rate": 0.0,
                        "min_due_amount": 0.0,
                        "payment_made_amount": 0.0,
                        "rewards_points_balance": 0,
                        "card_status": cc["card_status"],
                        "delinquency_flag": False,
                    })

            for ln in cust_loans:
                loan_id = ln["loan_id"]
                ln_state = running_loans[loan_id]
                outstanding = ln_state["outstanding_balance"]
                
                interest_rate = ln["interest_rate"]
                emi = ln["emi_amount"]
                interest = outstanding * (interest_rate / 12.0 / 100.0)
                principal = emi - interest
                principal = min(principal, outstanding)
                
                is_delinquent = ln_state["dpd_days"] > 0

                loan_snapshots.append({
                    "loan_id": loan_id,
                    "snapshot_month": snapshot_month,
                    "outstanding_balance": float(round(outstanding, 2)),
                    "emi_amount": float(emi),
                    "dpd_days": int(ln_state["dpd_days"]),
                    "overdue_amount": float(round(emi if is_delinquent else 0.0, 2)),
                    "principal_paid_amount": float(round(0.0 if is_delinquent else principal, 2)),
                    "interest_paid_amount": float(round(0.0 if is_delinquent else interest, 2)),
                    "installment_due_amount": float(emi),
                    "installment_paid_amount": float(round(0.0 if is_delinquent else emi, 2)),
                    "loan_status": ln_state["status"],
                    "restructuring_flag": False,
                })

            holdings = product_holdings[cid]
            product_holdings_snapshots.append({
                "customer_id": cid,
                "snapshot_month": snapshot_month,
                "savings_account_flag": holdings["savings_account_flag"],
                "current_account_flag": holdings["current_account_flag"],
                "credit_card_flag": holdings["credit_card_flag"],
                "personal_loan_flag": holdings["personal_loan_flag"],
                "home_loan_flag": holdings["home_loan_flag"],
                "fixed_deposit_flag": holdings["fixed_deposit_flag"],
                "insurance_flag": holdings["insurance_flag"],
                "mutual_fund_flag": holdings["mutual_fund_flag"],
                "demat_account_flag": holdings["demat_account_flag"],
                "wealth_management_flag": holdings["wealth_management_flag"],
                "products_count": int(sum(holdings.values())),
            })

            curr_prod_cnt = sum(holdings.values())
            products_count_drop[cid] = curr_prod_cnt < previous_products_count[cid]
            previous_products_count[cid] = curr_prod_cnt

            recent_service_failures[cid] = "bank_service_failure" in events
            if recent_service_failures[cid]:
                service_failures_2m[cid] = min(2, service_failures_2m[cid] + 1)
            else:
                service_failures_2m[cid] = max(0, service_failures_2m[cid] - 1)

        all_transactions.extend(monthly_txns)

        # Step 4: Batch Generate Activity & Digital Engagement Snapshots
        active_cids_list = [c["customer_id"] for c in active_customers]
        active_personas_list = [c["persona"] for c in active_customers]

        monthly_activity_df = generate_monthly_activity(
            active_cids_list,
            snapshot_month,
            active_personas_list,
            rng,
            month_aggregates,
            {cid: list(customer_events[cid]) for cid in active_cids_list}
        )
        activity_snapshots.append(monthly_activity_df)

        login_counts_dict = {
            row["customer_id"]: row["login_count"]
            for row in monthly_activity_df.to_dicts()
        }
        dsl_dict = {
            row["customer_id"]: row["days_since_last_login"]
            for row in monthly_activity_df.to_dicts()
        }

        monthly_digital_df = generate_monthly_digital(
            active_cids_list,
            snapshot_month,
            active_personas_list,
            rng,
            login_counts_dict,
            dsl_dict,
            {cid: list(customer_events[cid]) for cid in active_cids_list}
        )
        digital_snapshots.append(monthly_digital_df)

        for row in monthly_digital_df.to_dicts():
            cid = row["customer_id"]
            if not row["mobile_app_active"] and row["web_sessions"] == 0:
                digital_inactive_months[cid] += 1
            else:
                digital_inactive_months[cid] = 0

        # Step 5: Complaints and Feedback
        new_comps = generate_complaints_for_month(
            active_customers,
            customer_events,
            snapshot_month,
            next_complaint_id,
            rng
        )
        all_complaints.extend(new_comps)
        # Update dynamic complaints index
        for comp in new_comps:
            complaints_by_customer.setdefault(comp["customer_id"], []).append(comp)
        next_complaint_id += len(new_comps)

        resolved_cids = {
            cid for cid, evts in customer_events.items()
            if "complaint_resolved" in evts
        }
        resolve_complaints_for_month(all_complaints, resolved_cids, snapshot_month, rng)

        unresolved_counts = {}
        for comp in all_complaints:
            cid = comp["customer_id"]
            if not comp["resolved_flag"]:
                unresolved_counts[cid] = unresolved_counts.get(cid, 0) + 1
            elif comp["resolved_flag"] and comp["resolution_days"] is not None:
                if comp["customer_id"] in resolved_cids:
                    resolved_complaints_this_month.add(cid)

        new_feed = generate_feedback_for_month(
            active_customers,
            resolved_complaints_this_month,
            unresolved_counts,
            customer_events,
            snapshot_month,
            next_feedback_id,
            rng
        )
        all_feedback.extend(new_feed)
        next_feedback_id += len(new_feed)

        # Step 6: Churn Scoring
        m_minus_6 = add_months(snapshot_month, -6)
        comp_count_6m = {}
        for comp in all_complaints:
            cid = comp["customer_id"]
            if comp["complaint_month"] >= m_minus_6 and comp["complaint_month"] <= snapshot_month:
                comp_count_6m[cid] = comp_count_6m.get(cid, 0) + 1

        for c in active_customers:
            cid = c["customer_id"]
            persona = Persona(c["persona"])

            ln_dpd = 0
            ln_status = "Active"
            cust_lns = [running_loans[l["loan_id"]] for l in loans_by_customer.get(cid, [])]
            if cust_lns:
                ln_dpd = max(l["dpd_days"] for l in cust_lns)
                if any(l["status"] == "Delinquent" for l in cust_lns):
                    ln_status = "Delinquent"

            open_comps = unresolved_counts.get(cid, 0)
            cc_6m = comp_count_6m.get(cid, 0)

            cust_acc_ids = [a["account_id"] for a in accounts_by_customer.get(cid, [])]
            curr_bal = sum(running_balances[aid] for aid in cust_acc_ids)

            dig_inactive = digital_inactive_months[cid]
            failures_2m = service_failures_2m[cid]

            p_config = PERSONA_CONFIGS[persona]
            base_prob = rng.uniform(p_config.base_monthly_churn_min, p_config.base_monthly_churn_max)
            base_prob = np.clip(base_prob, 1e-5, 1 - 1e-5)
            base_rate = float(np.log(base_prob / (1 - base_prob)))

            c_input = ChurnInput(
                persona=persona,
                base_rate=base_rate,
                event_score=float(len(customer_events[cid]) * 0.1),
                trend_score=0.0,
                product_score=float((10 - sum(product_holdings[cid].values())) / 10.0),
                complaint_score=float((cc_6m * 0.15) + (open_comps * 0.25)),
                loan_stress_score=float(ln_dpd / 90.0),
                digital_inactivity_score=float(dig_inactive * 0.2),
                dpd_days=ln_dpd,
                loan_status=ln_status,
                complaint_count_6m=cc_6m,
                unresolved_complaints=open_comps,
                current_balance=curr_bal,
                months_without_salary=months_without_salary[cid],
                service_failures_2m=failures_2m,
                digital_inactive_months=dig_inactive,
                core_account_closed=False,
                recent_salary_job_change="salary_job_change" in customer_events[cid],
                products_count_drop=products_count_drop[cid],
                recent_service_failure=recent_service_failures[cid]
            )

            c_res = calculate_churn(c_input, rng, use_threshold=False)

            if c_res.churned:
                state = spine_state[cid]
                state["churned_flag"] = True
                state["churn_month"] = snapshot_month
                state["churn_reason"] = c_res.churn_reason

            spine_state[cid]["active_months_generated"] += 1

        if streaming and output_dir:
            from pipeline.writer import write_to_parquet
            m_acc_snap = pl.DataFrame(account_snapshots)
            m_card_snap = pl.DataFrame(card_snapshots)
            m_loan_snap = pl.DataFrame(loan_snapshots)
            m_holdings_snap = pl.DataFrame(product_holdings_snapshots)
            m_txns_snap = pl.DataFrame(all_transactions)
            m_activity_snap = pl.concat(activity_snapshots) if activity_snapshots else pl.DataFrame()
            m_digital_snap = pl.concat(digital_snapshots) if digital_snapshots else pl.DataFrame()

            temp_dfs = {
                "account_monthly_snapshot": m_acc_snap,
                "card_monthly_snapshot": m_card_snap,
                "loan_monthly_snapshot": m_loan_snap,
                "product_holdings_monthly": m_holdings_snap,
                "transaction_fact": m_txns_snap,
                "customer_monthly_activity": m_activity_snap,
                "digital_engagement_monthly": m_digital_snap,
            }
            write_to_parquet(temp_dfs, output_dir)

            all_transactions = []
            account_snapshots = []
            card_snapshots = []
            loan_snapshots = []
            product_holdings_snapshots = []
            activity_snapshots = []
            digital_snapshots = []

    # Final compilations
    if dynamic_events_fired:
        dynamic_events_df = pl.DataFrame(
            dynamic_events_fired,
            schema={
                "customer_id": pl.Int64,
                "event_month": pl.Date,
                "event_type": pl.String,
                "is_fired": pl.Boolean,
            }
        )
        final_events_df = pl.concat([spine.scheduled_events, dynamic_events_df]).sort(["customer_id", "event_month"])
    else:
        final_events_df = spine.scheduled_events

    final_state_df = pl.DataFrame(
        [
            {
                "customer_id": cid,
                "persona": st["persona"],
                "low_sensitivity_segment": st["low_sensitivity_segment"],
                "churn_month": st["churn_month"],
                "churned_flag": st["churned_flag"],
                "churn_reason": st["churn_reason"],
                "active_months_generated": st["active_months_generated"],
            }
            for cid, st in spine_state.items()
        ],
        schema={
            "customer_id": pl.Int64,
            "persona": pl.String,
            "low_sensitivity_segment": pl.Boolean,
            "churn_month": pl.Date,
            "churned_flag": pl.Boolean,
            "churn_reason": pl.String,
            "active_months_generated": pl.Int32,
        }
    )

    complaints_df = pl.DataFrame(
        all_complaints,
        schema={
            "complaint_id": pl.Int64,
            "customer_id": pl.Int64,
            "complaint_date": pl.Date,
            "complaint_month": pl.Date,
            "channel": pl.String,
            "category": pl.String,
            "severity": pl.String,
            "resolution_days": pl.Int32,
            "resolved_flag": pl.Boolean,
            "escalated_flag": pl.Boolean,
            "csat_score": pl.Int32,
            "root_cause": pl.String,
            "status": pl.String,
        }
    )

    feedback_df = pl.DataFrame(
        all_feedback,
        schema={
            "feedback_id": pl.Int64,
            "customer_id": pl.Int64,
            "feedback_date": pl.Date,
            "feedback_month": pl.Date,
            "survey_channel": pl.String,
            "survey_topic": pl.String,
            "nps_score": pl.Int32,
            "csat_score": pl.Int32,
        }
    )

    customer_master_df = customer_df
    account_master_df = pl.DataFrame(accounts, schema=accounts_df.schema)
    card_portfolio_df = pl.DataFrame(cards, schema=cards_df.schema)
    loan_master_df = pl.DataFrame(loans, schema=loans_df.schema)

    labels_df = generate_churn_labels(final_state_df, config)

    if streaming and output_dir:
        account_snapshots_df = pl.read_parquet(os.path.join(output_dir, "account_monthly_snapshot"))
        card_snapshots_df = pl.read_parquet(os.path.join(output_dir, "card_monthly_snapshot"))
        loan_snapshots_df = pl.read_parquet(os.path.join(output_dir, "loan_monthly_snapshot"))
        holdings_df = pl.read_parquet(os.path.join(output_dir, "product_holdings_monthly"))
        activity_df = pl.read_parquet(os.path.join(output_dir, "customer_monthly_activity"))
        digital_df = pl.read_parquet(os.path.join(output_dir, "digital_engagement_monthly"))
    else:
        account_snapshots_df = pl.DataFrame(account_snapshots)
        card_snapshots_df = pl.DataFrame(card_snapshots)
        loan_snapshots_df = pl.DataFrame(loan_snapshots)
        holdings_df = pl.DataFrame(product_holdings_snapshots)
        activity_df = pl.concat(activity_snapshots) if activity_snapshots else pl.DataFrame()
        digital_df = pl.concat(digital_snapshots) if digital_snapshots else pl.DataFrame()

    features_df = generate_feature_snapshots(
        customer_master_df,
        account_master_df,
        account_snapshots_df,
        card_portfolio_df,
        card_snapshots_df,
        loan_master_df,
        loan_snapshots_df,
        holdings_df,
        activity_df,
        digital_df,
        complaints_df,
        feedback_df,
        labels_df,
        config
    )

    res_dict = {
        "branch_master": branches_df,
        "customer_master": customer_master_df,
        "account_master": account_master_df,
        "card_portfolio": card_portfolio_df,
        "loan_master": loan_master_df,
        "churn_simulation_state": final_state_df,
        "customer_complaints": complaints_df,
        "customer_feedback": feedback_df,
        "customer_churn_label": labels_df,
        "churn_feature_snapshot": features_df,
    }

    if not streaming:
        res_dict.update({
            "account_monthly_snapshot": account_snapshots_df,
            "card_monthly_snapshot": card_snapshots_df,
            "loan_monthly_snapshot": loan_snapshots_df,
            "product_holdings_monthly": holdings_df,
            "transaction_fact": pl.DataFrame(all_transactions) if all_transactions else pl.DataFrame(),
            "customer_monthly_activity": activity_df,
            "digital_engagement_monthly": digital_df,
        })

    return res_dict
