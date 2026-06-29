"""Feedback survey generator for the synthetic retail banking universe.

Simulates customer feedback surveys (NPS and CSAT) based on active customer state,
unresolved complaints, and service failures. Enforces survey limits and deduplication.
"""

from datetime import date
from typing import List, Dict, Any, Set
import numpy as np

from config.personas import Persona

SURVEY_CHANNELS = ["SMS", "Email", "In-App Push"]


def generate_feedback_for_month(
    active_customers: List[Dict[str, Any]],
    resolved_complaints_this_month: Set[int],  # Customer IDs with resolved complaints
    unresolved_complaints_counts: Dict[
        int, int
    ],  # Customer ID -> open complaints count
    customer_events: Dict[int, Set[str]],
    snapshot_month: date,
    current_max_id: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Generates feedback surveys for the current simulation month.

    Enforces the rules:
    - 20% CSAT for complaint-resolved customers (Topic: "Complaint Resolution")
    - 2% random CSAT/NPS for other active customers (Topic: "General" or "Digital Experience")
    - Deduplicates to at most 1 survey per customer per month.

    Args:
        active_customers: List of active customer dicts.
        resolved_complaints_this_month: Set of customer IDs that had a complaint resolved this month.
        unresolved_complaints_counts: Dict of customer ID -> number of unresolved complaints.
        customer_events: Dict of customer ID -> set of active event names this month.
        snapshot_month: Current simulation month.
        current_max_id: Starting feedback_id.
        rng: Seeded numpy random generator.

    Returns:
        List[Dict[str, Any]]: Generated feedback records.
    """
    surveys = []
    feedback_id = current_max_id

    # Determine month length for random survey date
    if snapshot_month.month == 12:
        next_month = date(snapshot_month.year + 1, 1, 1)
    else:
        next_month = date(snapshot_month.year, snapshot_month.month + 1, 1)
    days_in_month = (next_month - snapshot_month).days

    # We will identify which customers trigger each survey and then deduplicate
    complaint_survey_custs = set()
    random_survey_custs = set()

    for cust in active_customers:
        cid = cust["customer_id"]
        # Trigger 1: Resolved complaint survey (20% sample)
        if cid in resolved_complaints_this_month:
            if rng.random() < 0.20:
                complaint_survey_custs.add(cid)

    for cust in active_customers:
        cid = cust["customer_id"]
        # Trigger 2: Random 2% sample of OTHER active customers
        if cid not in complaint_survey_custs:
            if rng.random() < 0.02:
                random_survey_custs.add(cid)

    # Union the two groups for processing (already deduplicated because they are disjoint)
    survey_targets = []
    for cid in complaint_survey_custs:
        survey_targets.append((cid, "Complaint Resolution"))
    for cid in random_survey_custs:
        survey_targets.append((cid, rng.choice(["General", "Digital Experience"])))

    # Convert customer list to dictionary for fast persona lookup
    cust_lookup = {c["customer_id"]: c for c in active_customers}

    for cid, topic in survey_targets:
        cust = cust_lookup.get(cid)
        if not cust:
            continue

        persona = Persona(cust["persona"])
        events = customer_events.get(cid, set())
        open_complaints = unresolved_complaints_counts.get(cid, 0)

        # Baseline scores by persona
        if persona == Persona.AFFLUENT_MULTI_PRODUCT:
            base_nps = 8.5
            base_csat = 4.2
        elif persona == Persona.SALARY_CORE:
            base_nps = 8.0
            base_csat = 4.0
        elif persona == Persona.DIGITAL_NATIVE:
            base_nps = 8.2
            base_csat = 4.1
        elif persona == Persona.DORMANT_WEALTHY:
            base_nps = 7.0
            base_csat = 3.5
        elif persona == Persona.CREDIT_STRESSED:
            base_nps = 6.0
            base_csat = 3.0
        else:  # COMPLAINT_PRONE_CHURNER
            base_nps = 5.0
            base_csat = 2.5

        # Modify scores based on customer happiness status
        nps_mod = 0.0
        csat_mod = 0.0

        if open_complaints > 0:
            nps_mod -= min(4.0, open_complaints * 1.5)
            csat_mod -= min(2.0, open_complaints * 0.7)

        if "bank_service_failure" in events:
            nps_mod -= 3.5
            csat_mod -= 1.8

        if "fee_hike_or_service_charge" in events:
            nps_mod -= 1.5
            csat_mod -= 0.8

        # If it is complaint-resolved, and CSAT was recorded on the complaint itself,
        # we can align them. Otherwise, resolved complaints increase satisfaction slightly.
        if topic == "Complaint Resolution":
            nps_mod += 1.0
            csat_mod += 0.5

        # Sample score with noise
        nps = base_nps + nps_mod + rng.normal(0, 1.0)
        csat = base_csat + csat_mod + rng.normal(0, 0.5)

        # Force bounds
        nps_score = int(np.clip(round(nps), 0, 10))
        csat_score = int(np.clip(round(csat), 1, 5))

        # Generate survey date
        day = rng.integers(1, days_in_month + 1)
        survey_date = date(snapshot_month.year, snapshot_month.month, int(day))

        surveys.append(
            {
                "feedback_id": feedback_id,
                "customer_id": cid,
                "feedback_date": survey_date,
                "feedback_month": snapshot_month,
                "survey_channel": rng.choice(SURVEY_CHANNELS),
                "survey_topic": topic,
                "nps_score": nps_score,
                "csat_score": csat_score,
            }
        )
        feedback_id += 1

    return surveys
