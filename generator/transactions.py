"""Transaction helpers library for the synthetic retail banking universe.

Defines functional helpers used by the monthly simulation loop to generate
individual transaction records (credits, debits, fees) dynamically.
"""

from datetime import date, datetime, time
from typing import List, Optional, Dict, Set
import numpy as np

from config.personas import Persona, PERSONA_CONFIGS


# Indian merchant details for realistic transaction generation
MERCHANTS = {
    "Groceries": ["BigBasket", "Zepto", "Blinkit", "DMart", "Reliance Smart"],
    "Dining": ["Zomato", "Swiggy", "KFC", "Starbucks", "Barbeque Nation"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Reliance Digital", "Zudio"],
    "Utilities": ["Bescom", "Tata Power", "Airtel Pay", "Jio Recharge", "Adani Gas"],
    "Fuel": ["HPCL Station", "BPCL Petrol Pump", "IOCL Outlet"],
    "Travel": ["MakeMyTrip", "Ola Cabs", "Uber Rides", "IRCTC", "IndiGo"],
}


def generate_salary_credit(
    transaction_id: int,
    customer_id: int,
    account_id: int,
    amount: float,
    txn_date: date,
    city: str,
    state: str,
) -> dict:
    """Generates a salary credit transaction record.

    Args:
        transaction_id: Unique identifier for the transaction.
        customer_id: Customer identifier.
        account_id: Account identifier.
        amount: Salary credit amount.
        txn_date: Date of transaction.
        city: Customer's city.
        state: Customer's state.

    Returns:
        dict: The transaction fact record.
    """
    timestamp = datetime.combine(txn_date, time(9, 30, 0))  # Standard morning deposit

    return {
        "transaction_id": transaction_id,
        "account_id": account_id,
        "customer_id": customer_id,
        "txn_timestamp": timestamp,
        "txn_date": txn_date,
        "txn_month": date(txn_date.year, txn_date.month, 1),
        "txn_type": "Salary Credit",
        "direction": "Credit",
        "channel": "Transfer",
        "amount": round(float(amount), 2),
        "currency": "INR",
        "merchant_category": "Payroll",
        "merchant_name": "Employer Corp",
        "counterparty_type": "Corporate",
        "city": city,
        "state": state,
        "is_salary_credit": True,
        "is_fee": False,
        "is_reversal": False,
        "balance_after_txn": 0.0,  # Will be updated by simulation loop
    }


def generate_non_salary_income(
    start_txn_id: int,
    customer_id: int,
    account_id: int,
    txn_month: date,
    persona: Persona,
    city: str,
    state: str,
    rng: np.random.Generator,
) -> List[dict]:
    """Models irregular income for non-salary account holders.

    Returns 1-3 credit transactions per month drawn from persona income distribution.
    """
    p_config = PERSONA_CONFIGS[persona]
    
    # Draw monthly income from lognormal (same params, divide by 12)
    drawn_annual = rng.lognormal(p_config.income_log_mu, p_config.income_log_sigma)
    drawn_annual = np.clip(drawn_annual, p_config.income_clip_min, p_config.income_clip_max)
    monthly_income = drawn_annual / 12.0
    
    # 1. Dormant Wealthy: Quarterly credits (investment returns/rental)
    if persona == Persona.DORMANT_WEALTHY:
        # Check if it is the 3rd month (e.g. March, June, September, December)
        if txn_month.month % 3 == 0:
            n_credits = 1
            monthly_income *= 3.0
        else:
            return []
    else:
        # Number of credit events varies by persona
        n_credits = {
            Persona.DIGITAL_NATIVE: int(rng.integers(2, 5)),      # freelance/gig, multiple small credits
            Persona.AFFLUENT_MULTI_PRODUCT: int(rng.integers(1, 3)),  # business income, fewer larger credits  
            Persona.CREDIT_STRESSED: int(rng.integers(1, 3)),     # irregular, sometimes partial
            Persona.COMPLAINT_PRONE_CHURNER: int(rng.integers(1, 3)),
            Persona.SALARY_CORE: 1,  # fallback
        }[persona]

    # For credit_stressed, 15% monthly probability of a "short month" where total credits are only 60-80%
    if persona == Persona.CREDIT_STRESSED and rng.random() < 0.15:
        monthly_income *= rng.uniform(0.60, 0.80)

    # Split monthly income across n_credits transactions using random weights
    if n_credits > 1:
        weights = rng.random(n_credits)
        weights /= weights.sum()
    else:
        weights = [1.0]

    # Find number of days in the month
    if txn_month.month == 12:
        next_month = date(txn_month.year + 1, 1, 1)
    else:
        next_month = date(txn_month.year, txn_month.month + 1, 1)
    days_in_month = (next_month - txn_month).days

    # Select random days for credits
    credit_days = rng.choice(np.arange(1, days_in_month + 1), size=n_credits, replace=True)

    txns = []
    current_id = start_txn_id

    for i, w in enumerate(weights):
        day = int(credit_days[i])
        amt = round(monthly_income * w, 2)
        txn_date = date(txn_month.year, txn_month.month, day)
        timestamp = datetime.combine(txn_date, time(11, 0, 0))  # Standard morning credit

        if persona == Persona.DORMANT_WEALTHY:
            cp_type = "Self Transfer"
            merchant_name = "Self Account"
            category = "Investment Income"
        else:
            cp_type = "Business Income"
            merchant_name = rng.choice(["Freelance Client", "Vendor Transfer", "Customer Payment", "Consulting Fee"])
            category = "Professional Services"

        txns.append(
            {
                "transaction_id": current_id,
                "account_id": account_id,
                "customer_id": customer_id,
                "txn_timestamp": timestamp,
                "txn_date": txn_date,
                "txn_month": txn_month,
                "txn_type": "Direct Credit",
                "direction": "Credit",
                "channel": rng.choice(["UPI", "Transfer", "Internet Banking"], p=[0.5, 0.3, 0.2]),
                "amount": amt,
                "currency": "INR",
                "merchant_category": category,
                "merchant_name": merchant_name,
                "counterparty_type": cp_type,
                "city": city,
                "state": state,
                "is_salary_credit": False,
                "is_fee": False,
                "is_reversal": False,
                "balance_after_txn": 0.0,
            }
        )
        current_id += 1

    return txns


def generate_regular_transactions(
    start_txn_id: int,
    customer_id: int,
    account_id: int,
    txn_date: date,
    persona: Persona,
    city: str,
    state: str,
    rng: np.random.Generator,
    active_events: Optional[List[str]] = None,
) -> List[dict]:
    """Generates a set of regular debit transactions for a customer on a given day.

    Number of transactions and amounts are adjusted by persona and active events.

    Args:
        start_txn_id: Starting unique transaction ID.
        customer_id: Customer identifier.
        account_id: Account identifier.
        txn_date: Date of transactions.
        persona: Persona of the customer.
        city: City of the transaction.
        state: State of the transaction.
        rng: Seeded numpy random generator.
        active_events: List of active HiddenEvent names.

    Returns:
        List[dict]: List of transaction fact records.
    """
    if active_events is None:
        active_events = []

    # 1. Determine transaction count based on persona and events
    # Digital natives transact frequently; dormant wealthy rarely do.
    if persona == Persona.DIGITAL_NATIVE:
        base_count_lambda = 1.2
    elif persona == Persona.DORMANT_WEALTHY:
        base_count_lambda = 0.1
    elif persona == Persona.AFFLUENT_MULTI_PRODUCT:
        base_count_lambda = 0.8
    else:
        base_count_lambda = 0.5

    # Events affect transaction volume
    if "large_life_expense" in active_events:
        base_count_lambda += 1.5
    if "salary_job_change" in active_events:
        base_count_lambda += 0.3

    txn_count = rng.poisson(base_count_lambda)
    if txn_count == 0:
        return []

    # Channels and merchant category preferences
    channels = ["UPI", "POS", "ATM", "Internet Banking", "Mobile App"]
    # Digital native prefers UPI; dormant wealthy prefers ATM or Branch
    if persona == Persona.DIGITAL_NATIVE:
        channel_probs = [0.70, 0.15, 0.02, 0.03, 0.10]
    elif persona == Persona.DORMANT_WEALTHY:
        channel_probs = [0.10, 0.30, 0.40, 0.15, 0.05]
    else:
        channel_probs = [0.45, 0.25, 0.15, 0.05, 0.10]

    txns = []
    current_id = start_txn_id

    for _ in range(txn_count):
        category = rng.choice(list(MERCHANTS.keys()))
        merchant = rng.choice(MERCHANTS[category])
        channel = rng.choice(channels, p=channel_probs)

        # 2. Determine amount based on category and events
        if category == "Groceries":
            amt = rng.uniform(200.0, 2500.0)
        elif category == "Dining":
            amt = rng.uniform(300.0, 4000.0)
        elif category == "Shopping":
            amt = rng.uniform(500.0, 8000.0)
        elif category == "Utilities":
            amt = rng.uniform(100.0, 3000.0)
        elif category == "Fuel":
            amt = rng.uniform(500.0, 3000.0)
        else:  # Travel
            amt = rng.uniform(1000.0, 15000.0)

        # Large life expense scaling
        if "large_life_expense" in active_events and rng.random() < 0.5:
            amt *= rng.uniform(5.0, 15.0)

        # Volatility multiplier for job change
        if "salary_job_change" in active_events:
            amt *= rng.uniform(0.8, 2.0)

        # Round to 2 decimal places
        amt = round(float(amt), 2)

        # Generate transaction hour
        hour = rng.integers(7, 23)
        minute = rng.integers(0, 60)
        second = rng.integers(0, 60)
        timestamp = datetime.combine(txn_date, time(hour, minute, second))

        txns.append(
            {
                "transaction_id": current_id,
                "account_id": account_id,
                "customer_id": customer_id,
                "txn_timestamp": timestamp,
                "txn_date": txn_date,
                "txn_month": date(txn_date.year, txn_date.month, 1),
                "txn_type": "Debit Card / UPI",
                "direction": "Debit",
                "channel": channel,
                "amount": amt,
                "currency": "INR",
                "merchant_category": category,
                "merchant_name": merchant,
                "counterparty_type": "Merchant",
                "city": city,
                "state": state,
                "is_salary_credit": False,
                "is_fee": False,
                "is_reversal": False,
                "balance_after_txn": 0.0,
            }
        )
        current_id += 1

    return txns


def generate_fee_or_charge(
    transaction_id: int,
    customer_id: int,
    account_id: int,
    txn_date: date,
    amount: float,
    city: str,
    state: str,
) -> dict:
    """Generates a fee transaction record.

    Args:
        transaction_id: Unique identifier for the transaction.
        customer_id: Customer identifier.
        account_id: Account identifier.
        txn_date: Date of transaction.
        amount: Fee amount.
        city: City of the transaction.
        state: State of the transaction.

    Returns:
        dict: The transaction fact record.
    """
    timestamp = datetime.combine(txn_date, time(23, 59, 59))  # End of day charge

    return {
        "transaction_id": transaction_id,
        "account_id": account_id,
        "customer_id": customer_id,
        "txn_timestamp": timestamp,
        "txn_date": txn_date,
        "txn_month": date(txn_date.year, txn_date.month, 1),
        "txn_type": "Fee Charged",
        "direction": "Debit",
        "channel": "System",
        "amount": round(float(amount), 2),
        "currency": "INR",
        "merchant_category": "Service Charges",
        "merchant_name": "Bank Service Fee",
        "counterparty_type": "Bank",
        "city": city,
        "state": state,
        "is_salary_credit": False,
        "is_fee": True,
        "is_reversal": False,
        "balance_after_txn": 0.0,
    }


def generate_monthly_regular_transactions(
    active_customers: List[dict],
    customer_events: Dict[int, Set[str]],
    snapshot_month: date,
    start_txn_id: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Generates all regular debit transactions for all active customers in a single month using vectorized operations."""
    if snapshot_month.month == 12:
        next_month = date(snapshot_month.year + 1, 1, 1)
    else:
        next_month = date(snapshot_month.year, snapshot_month.month + 1, 1)
    days_in_month = (next_month - snapshot_month).days

    cids = []
    personas = []
    cities = []
    states = []
    lambda_days_list = []
    base_lambdas = []
    
    for c in active_customers:
        cid = c["customer_id"]
        persona = Persona(c["persona"])
        cids.append(cid)
        personas.append(persona)
        cities.append(c["city"])
        states.append(c["state"])
        
        if persona == Persona.DIGITAL_NATIVE:
            ld = 10
            bl = 1.2
        elif persona == Persona.DORMANT_WEALTHY:
            ld = 2
            bl = 0.1
        elif persona == Persona.AFFLUENT_MULTI_PRODUCT:
            ld = 5
            bl = 0.8
        else:
            ld = 5
            bl = 0.5
            
        lambda_days_list.append(ld)
        base_lambdas.append(bl)

    n_cust = len(cids)
    if n_cust == 0:
        return []

    num_days = rng.poisson(lambda_days_list)
    num_days = np.clip(num_days, 1, days_in_month)

    days_arr = np.arange(1, days_in_month + 1)
    cust_days = []
    for nd in num_days:
        cust_days.append(rng.choice(days_arr, size=nd, replace=False))

    cids_repeated = []
    days_repeated = []
    lambdas_repeated = []
    indices_repeated = []
    
    for i, (cid, days) in enumerate(zip(cids, cust_days)):
        nd = len(days)
        if nd > 0:
            c_events = customer_events.get(cid, set())
            bl = base_lambdas[i]
            if "large_life_expense" in c_events:
                bl += 1.5
            if "salary_job_change" in c_events:
                bl += 0.3
            cids_repeated.append(np.full(nd, cid, dtype=np.int64))
            days_repeated.append(days)
            lambdas_repeated.append(np.full(nd, bl))
            indices_repeated.append(np.full(nd, i, dtype=np.int32))

    if not cids_repeated:
        return []

    cids_cd = np.concatenate(cids_repeated)
    days_cd = np.concatenate(days_repeated)
    lambdas_cd = np.concatenate(lambdas_repeated)
    indices_cd = np.concatenate(indices_repeated)

    counts_cd = rng.poisson(lambdas_cd)
    valid_cd = counts_cd > 0
    
    if not np.any(valid_cd):
        return []

    cids_cd = cids_cd[valid_cd]
    days_cd = days_cd[valid_cd]
    counts_cd = counts_cd[valid_cd]
    indices_cd = indices_cd[valid_cd]

    txn_cids = np.repeat(cids_cd, counts_cd)
    txn_days = np.repeat(days_cd, counts_cd)
    txn_indices = np.repeat(indices_cd, counts_cd)
    n_txns = len(txn_cids)

    cat_keys = list(MERCHANTS.keys())
    txn_cat_idx = rng.integers(0, len(cat_keys), size=n_txns)
    
    txn_merchants = []
    txn_cats = []
    for idx in txn_cat_idx:
        cat = cat_keys[idx]
        txn_cats.append(cat)
        merchants_list = MERCHANTS[cat]
        m_idx = rng.integers(0, len(merchants_list))
        txn_merchants.append(merchants_list[m_idx])

    channels = ["UPI", "POS", "ATM", "Internet Banking", "Mobile App"]
    txn_channels = [None] * n_txns
    
    for p_val in [Persona.DIGITAL_NATIVE, Persona.DORMANT_WEALTHY, None]:
        if p_val == Persona.DIGITAL_NATIVE:
            p_mask = np.array([personas[idx] == Persona.DIGITAL_NATIVE for idx in txn_indices])
            p_probs = [0.70, 0.15, 0.02, 0.03, 0.10]
        elif p_val == Persona.DORMANT_WEALTHY:
            p_mask = np.array([personas[idx] == Persona.DORMANT_WEALTHY for idx in txn_indices])
            p_probs = [0.10, 0.30, 0.40, 0.15, 0.05]
        else:
            p_mask = np.array([personas[idx] != Persona.DIGITAL_NATIVE and personas[idx] != Persona.DORMANT_WEALTHY for idx in txn_indices])
            p_probs = [0.45, 0.25, 0.15, 0.05, 0.10]
            
        p_count = np.sum(p_mask)
        if p_count > 0:
            drawn_channels = rng.choice(channels, size=p_count, p=p_probs)
            p_indices = np.where(p_mask)[0]
            for idx, chan in zip(p_indices, drawn_channels):
                txn_channels[idx] = chan

    txn_amounts = np.zeros(n_txns)
    for cat in cat_keys:
        cat_mask = np.array([c == cat for c in txn_cats])
        cat_count = np.sum(cat_mask)
        if cat_count > 0:
            if cat == "Groceries":
                low, high = 200.0, 2500.0
            elif cat == "Dining":
                low, high = 300.0, 4000.0
            elif cat == "Shopping":
                low, high = 500.0, 8000.0
            elif cat == "Utilities":
                low, high = 100.0, 3000.0
            elif cat == "Fuel":
                low, high = 500.0, 3000.0
            else:
                low, high = 1000.0, 15000.0
                
            txn_amounts[cat_mask] = rng.uniform(low, high, size=cat_count)

    has_large_life_expense = np.array(["large_life_expense" in customer_events.get(cid, set()) for cid in txn_cids])
    if np.any(has_large_life_expense):
        n_lle = np.sum(has_large_life_expense)
        lle_draws = rng.random(n_lle) < 0.5
        lle_scales = np.ones(n_lle)
        if np.any(lle_draws):
            lle_scales[lle_draws] = rng.uniform(5.0, 15.0, size=np.sum(lle_draws))
        txn_amounts[has_large_life_expense] *= lle_scales

    has_job_change = np.array(["salary_job_change" in customer_events.get(cid, set()) for cid in txn_cids])
    if np.any(has_job_change):
        n_jc = np.sum(has_job_change)
        jc_scales = rng.uniform(0.8, 2.0, size=n_jc)
        txn_amounts[has_job_change] *= jc_scales

    hours = rng.integers(7, 23, size=n_txns)
    minutes = rng.integers(0, 60, size=n_txns)
    seconds = rng.integers(0, 60, size=n_txns)

    txns = []
    curr_id = start_txn_id
    
    for i in range(n_txns):
        cid = int(txn_cids[i])
        day = int(txn_days[i])
        idx = int(txn_indices[i])
        
        txn_date = date(snapshot_month.year, snapshot_month.month, day)
        timestamp = datetime.combine(txn_date, time(int(hours[i]), int(minutes[i]), int(seconds[i])))
        
        txns.append({
            "transaction_id": curr_id,
            "account_id": None,
            "customer_id": cid,
            "txn_timestamp": timestamp,
            "txn_date": txn_date,
            "txn_month": snapshot_month,
            "txn_type": "Debit Card / UPI",
            "direction": "Debit",
            "channel": txn_channels[i],
            "amount": round(float(txn_amounts[i]), 2),
            "currency": "INR",
            "merchant_category": txn_cats[i],
            "merchant_name": txn_merchants[i],
            "counterparty_type": "Merchant",
            "city": cities[idx],
            "state": states[idx],
            "is_salary_credit": False,
            "is_fee": False,
            "is_reversal": False,
            "balance_after_txn": 0.0,
        })
        curr_id += 1

    return txns


def generate_monthly_non_salary_income(
    non_salary_customers: List[dict],
    customer_events: Dict[int, Set[str]],
    snapshot_month: date,
    start_txn_id: int,
    rng: np.random.Generator,
) -> List[dict]:
    """Generates all irregular income credits for non-salary account holders in a single month using vectorized operations."""
    txns = []
    curr_id = start_txn_id
    
    if not non_salary_customers:
        return []

    if snapshot_month.month == 12:
        next_month = date(snapshot_month.year + 1, 1, 1)
    else:
        next_month = date(snapshot_month.year, snapshot_month.month + 1, 1)
    days_in_month = (next_month - snapshot_month).days

    cids = [c["customer_id"] for c in non_salary_customers]
    personas = [Persona(c["persona"]) for c in non_salary_customers]
    acc_ids = [c["account_id"] for c in non_salary_customers]
    cities = [c["city"] for c in non_salary_customers]
    states = [c["state"] for c in non_salary_customers]
    incomes = [c["annual_income"] for c in non_salary_customers]

    n_cust = len(cids)
    
    mus = np.array([PERSONA_CONFIGS[p].income_log_mu for p in personas])
    sigmas = np.array([PERSONA_CONFIGS[p].income_log_sigma for p in personas])
    clip_mins = np.array([PERSONA_CONFIGS[p].income_clip_min for p in personas])
    clip_maxs = np.array([PERSONA_CONFIGS[p].income_clip_max for p in personas])
    
    drawn_annual = rng.lognormal(mus, sigmas)
    drawn_annual = np.clip(drawn_annual, clip_mins, clip_maxs)
    monthly_incomes = drawn_annual / 12.0

    n_credits_list = []
    for i, p in enumerate(personas):
        if p == Persona.DORMANT_WEALTHY:
            if snapshot_month.month % 3 == 0:
                nc = 1
                monthly_incomes[i] *= 3.0
            else:
                nc = 0
        else:
            if p == Persona.DIGITAL_NATIVE:
                nc = rng.integers(2, 5)
            elif p == Persona.SALARY_CORE:
                nc = 1
            else:
                nc = rng.integers(1, 3)
                
        if p == Persona.CREDIT_STRESSED and rng.random() < 0.15:
            monthly_incomes[i] *= rng.uniform(0.60, 0.80)
            
        n_credits_list.append(int(nc))

    for i in range(n_cust):
        nc = n_credits_list[i]
        if nc == 0:
            continue
            
        cid = cids[i]
        acc_id = acc_ids[i]
        persona = personas[i]
        income = monthly_incomes[i]
        
        if nc > 1:
            weights = rng.random(nc)
            weights /= weights.sum()
        else:
            weights = [1.0]
            
        credit_days = rng.choice(np.arange(1, days_in_month + 1), size=nc, replace=True)
        
        for k in range(nc):
            day = int(credit_days[k])
            amt = round(float(income * weights[k]), 2)
            txn_date = date(snapshot_month.year, snapshot_month.month, day)
            timestamp = datetime.combine(txn_date, time(11, 0, 0))
            
            if persona == Persona.DORMANT_WEALTHY:
                cp_type = "Self Transfer"
                merchant_name = "Self Account"
                category = "Investment Income"
            else:
                cp_type = "Business Income"
                merchant_name = rng.choice(["Freelance Client", "Vendor Transfer", "Customer Payment", "Consulting Fee"])
                category = "Professional Services"
                
            txns.append({
                "transaction_id": curr_id,
                "account_id": acc_id,
                "customer_id": cid,
                "txn_timestamp": timestamp,
                "txn_date": txn_date,
                "txn_month": snapshot_month,
                "txn_type": "Direct Credit",
                "direction": "Credit",
                "channel": rng.choice(["UPI", "Transfer", "Internet Banking"], p=[0.5, 0.3, 0.2]),
                "amount": amt,
                "currency": "INR",
                "merchant_category": category,
                "merchant_name": merchant_name,
                "counterparty_type": cp_type,
                "city": cities[i],
                "state": states[i],
                "is_salary_credit": False,
                "is_fee": False,
                "is_reversal": False,
                "balance_after_txn": 0.0,
            })
            curr_id += 1
            
    return txns
