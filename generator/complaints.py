"""Complaints generator for the synthetic retail banking universe.

Handles monthly complaints generation based on persona profiles and active events,
and resolves complaints with a minimum 1-month lag.
"""

from datetime import date
from typing import List, Dict, Any, Set
import numpy as np

from config.personas import PERSONA_CONFIGS, Persona

# Channel and Category options
CHANNELS = ["Call Center", "Mobile App", "Branch", "Web"]
SEVERITIES = ["Low", "Medium", "High"]

DEFAULT_CATEGORIES = [
    "Billing",
    "Service",
    "Transaction",
    "Tech Issue",
    "Account Access",
]
DEFAULT_ROOT_CAUSES = [
    "Unclear Charge Description",
    "Staff Behavior",
    "UI Glitch",
    "Wrong Statement",
    "Slow Response Time",
]


def generate_complaints_for_month(
    active_customers: List[Dict[str, Any]],
    customer_events: Dict[int, Set[str]],
    snapshot_month: date,
    current_max_id: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Generates new complaints for active customers in the current simulation month.

    Args:
        active_customers: List of customer dicts active in the current month.
        customer_events: Dict mapping customer_id -> set of active event names in the current month.
        snapshot_month: Current simulation month (first day of the month).
        current_max_id: The next available complaint_id.
        rng: Seeded numpy random generator.

    Returns:
        List[Dict[str, Any]]: Generated complaint records.
    """
    new_complaints = []
    complaint_id = current_max_id

    # Find month length to assign random complaint date
    if snapshot_month.month == 12:
        next_month = date(snapshot_month.year + 1, 1, 1)
    else:
        next_month = date(snapshot_month.year, snapshot_month.month + 1, 1)
    days_in_month = (next_month - snapshot_month).days

    for cust in active_customers:
        cid = cust["customer_id"]
        persona = Persona(cust["persona"])
        events = customer_events.get(cid, set())

        # Determine complaint probability
        p_config = PERSONA_CONFIGS[persona]
        base_prob = rng.uniform(
            p_config.complaint_rate_min, p_config.complaint_rate_max
        )

        # Event-driven boosts
        event_boost = 0.0
        if "bank_service_failure" in events:
            event_boost += 0.45
        if "salary_delay" in events:
            event_boost += 0.25
        if "fee_hike_or_service_charge" in events:
            event_boost += 0.15

        prob = min(0.95, base_prob + event_boost)

        if rng.random() < prob:
            # Generate complaint
            day = rng.integers(1, days_in_month + 1)
            complaint_date = date(snapshot_month.year, snapshot_month.month, int(day))

            # Channel selection
            channel = rng.choice(CHANNELS)
            severity = rng.choice(SEVERITIES, p=[0.5, 0.3, 0.2])

            # Trigger-driven categories & root causes
            if "bank_service_failure" in events:
                category = "Tech Issue"
                root_cause = "System Outage"
                severity = "High"
            elif "salary_delay" in events:
                category = "Transaction"
                root_cause = "Delayed Salary Processing"
                severity = rng.choice(["High", "Medium"])
            elif "fee_hike_or_service_charge" in events:
                category = "Billing"
                root_cause = "Excessive Charges"
                severity = "Medium"
            else:
                category = rng.choice(DEFAULT_CATEGORIES)
                root_cause = rng.choice(DEFAULT_ROOT_CAUSES)

            new_complaints.append(
                {
                    "complaint_id": complaint_id,
                    "customer_id": cid,
                    "complaint_date": complaint_date,
                    "complaint_month": snapshot_month,
                    "channel": channel,
                    "category": category,
                    "severity": severity,
                    "resolution_days": None,
                    "resolved_flag": False,
                    "escalated_flag": bool(rng.random() < 0.1),
                    "csat_score": None,
                    "root_cause": root_cause,
                    "status": "Open",
                }
            )
            complaint_id += 1

    return new_complaints


SEVERITY_CSAT_PARAMS = {
    "Low": {"mean": 3.5, "std": 0.9},
    "Medium": {"mean": 2.8, "std": 1.0},
    "High": {"mean": 2.0, "std": 1.0},
}
SEVERITY_RESOLUTION_DAYS = {
    "Low": (7, 25),
    "Medium": (20, 45),
    "High": (35, 75),
}


def resolve_complaints_for_month(
    all_complaints: List[Dict[str, Any]],
    resolved_customer_ids: Set[int],
    snapshot_month: date,
    rng: np.random.Generator,
) -> None:
    """Resolves open complaints for customers where a complaint_resolved event is fired.

    Enforces that a complaint can only be resolved in a month strictly after it was filed (N+1 lag).

    Args:
        all_complaints: The global list of complaints (modified in place).
        resolved_customer_ids: Set of customer_ids that triggered a COMPLAINT_RESOLVED event this month.
        snapshot_month: The current simulation month.
        rng: Seeded numpy random generator.
    """
    for comp in all_complaints:
        # Check if customer has a resolution event and complaint is currently Open
        # AND check that the complaint was filed in a month STRICTLY before the current snapshot month (N+1 lag)
        if (
            comp["customer_id"] in resolved_customer_ids
            and comp["status"] == "Open"
            and comp["complaint_month"] < snapshot_month
        ):
            # Resolve the complaint
            comp["status"] = "Resolved"
            comp["resolved_flag"] = True

            sev = comp["severity"]

            # Determine resolution days based on severity
            low_days, high_days = SEVERITY_RESOLUTION_DAYS.get(sev, (15, 45))
            resolution_days = int(rng.integers(low_days, high_days + 1))
            comp["resolution_days"] = resolution_days

            # Determine CSAT score based on severity parameters
            params = SEVERITY_CSAT_PARAMS.get(sev, {"mean": 3.0, "std": 1.0})
            csat_val = rng.normal(params["mean"], params["std"])
            csat_score = int(np.clip(np.round(csat_val), 1, 5))
            comp["csat_score"] = csat_score
