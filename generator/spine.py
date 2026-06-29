"""Customer spine generator for the synthetic retail banking universe.

Defines the Spine dataclass and the generate_spine function, which allocates
personas, low-sensitivity segments, and pre-schedules unconditional events.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Set
import numpy as np
import polars as pl

from config.constants import CUSTOMER_ID_START, LOW_SENSITIVITY_SHARE_RANGE
from config.events import EVENT_PROBABILITIES, HiddenEvent, UNCONDITIONAL_EVENTS
from config.personas import Persona
from config.simulation import SimulationConfig


@dataclass
class Spine:
    """Spine data structure containing simulation states and pre-scheduled events."""

    simulation_state: pl.DataFrame
    scheduled_events: pl.DataFrame


def generate_spine(
    config: SimulationConfig, customer_id_start: int = CUSTOMER_ID_START
) -> Spine:
    """Generates the customer spine and schedules unconditional events.

    Args:
        config: The top-level simulation configuration.
        customer_id_start: The starting customer ID offset.

    Returns:
        Spine: Contains the simulation state and scheduled events DataFrames.
    """
    # Initialize the generator with the global simulation seed for reproducibility
    rng = np.random.default_rng(config.seed)

    # 1. Generate customer IDs
    customer_ids = np.arange(
        customer_id_start, customer_id_start + config.n_customers, dtype=np.int64
    )

    # 2. Distribute personas uniformly across customers
    personas = list(Persona)
    persona_values = [p.value for p in personas]
    customer_personas_drawn = rng.choice(persona_values, size=config.n_customers)
    customer_personas = [str(p) for p in customer_personas_drawn]

    # 3. Sample low-sensitivity segment share and assign flags
    low_sens_share = rng.uniform(
        LOW_SENSITIVITY_SHARE_RANGE[0], LOW_SENSITIVITY_SHARE_RANGE[1]
    )
    low_sens_flags = rng.random(size=config.n_customers) < low_sens_share

    # 4. Generate the dates list for the simulation duration
    months: List[date] = []
    curr = config.sim_start
    for _ in range(config.sim_months):
        months.append(curr)
        # Add 1 month manually to maintain zero-I/O pure datetime arithmetic
        m = curr.month
        y = curr.year
        if m == 12:
            curr = date(y + 1, 1, 1)
        else:
            curr = date(y, m + 1, 1)

    # 5. Pre-schedule unconditional events persona by persona
    # Dict mapping (customer_id, event_month) to a set of triggered HiddenEvents
    triggered_events_dict: Dict[tuple[int, date], Set[HiddenEvent]] = {}

    for p in personas:
        # Get indices of customers belonging to this persona
        indices = [i for i, val in enumerate(customer_personas) if val == p.value]
        if not indices:
            continue
        p_cust_ids = customer_ids[indices]

        # Evaluate probabilities of all unconditional events for this persona
        for event in UNCONDITIONAL_EVENTS:
            prob = EVENT_PROBABILITIES[p][event]
            if prob <= 0.0:
                continue

            # Draw random probability matrices for (customers, months)
            draws = rng.random(size=(len(p_cust_ids), config.sim_months))
            triggered_indices = np.argwhere(draws < prob)

            for cust_idx, month_idx in triggered_indices:
                cid = int(p_cust_ids[cust_idx])
                m_date = months[month_idx]
                key = (cid, m_date)

                if key not in triggered_events_dict:
                    triggered_events_dict[key] = set()
                triggered_events_dict[key].add(event)

    # 6. Apply mutual exclusion rules on scheduled events and collect rows
    event_rows = []
    for (cid, m_date), events in triggered_events_dict.items():
        # Mutual Exclusion Rule 1: Salary Category
        # Discard SALARY_DELAY if SALARY_JOB_CHANGE occurs in the same month
        if (
            HiddenEvent.SALARY_JOB_CHANGE in events
            and HiddenEvent.SALARY_DELAY in events
        ):
            events.remove(HiddenEvent.SALARY_DELAY)

        # Mutual Exclusion Rule 2: Expense Category
        # Discard LARGE_LIFE_EXPENSE if HOME_PURCHASE occurs in the same month
        if (
            HiddenEvent.HOME_PURCHASE in events
            and HiddenEvent.LARGE_LIFE_EXPENSE in events
        ):
            events.remove(HiddenEvent.LARGE_LIFE_EXPENSE)

        for event in events:
            event_rows.append(
                {
                    "customer_id": cid,
                    "event_month": m_date,
                    "event_type": event.value,
                    "is_fired": False,
                }
            )

    # 7. Construct DataFrames using Polars
    # Sort state by customer_id for clean indexing
    simulation_state_df = pl.DataFrame(
        {
            "customer_id": customer_ids,
            "persona": customer_personas,
            "low_sensitivity_segment": low_sens_flags,
            "churn_month": [None] * config.n_customers,
            "churned_flag": [False] * config.n_customers,
            "churn_reason": [None] * config.n_customers,
            "active_months_generated": [0] * config.n_customers,
        },
        schema={
            "customer_id": pl.Int64,
            "persona": pl.String,
            "low_sensitivity_segment": pl.Boolean,
            "churn_month": pl.Date,
            "churned_flag": pl.Boolean,
            "churn_reason": pl.String,
            "active_months_generated": pl.Int32,
        },
    )

    if event_rows:
        scheduled_events_df = pl.DataFrame(
            event_rows,
            schema={
                "customer_id": pl.Int64,
                "event_month": pl.Date,
                "event_type": pl.String,
                "is_fired": pl.Boolean,
            },
        ).sort(["customer_id", "event_month"])
    else:
        scheduled_events_df = pl.DataFrame(
            schema={
                "customer_id": pl.Int64,
                "event_month": pl.Date,
                "event_type": pl.String,
                "is_fired": pl.Boolean,
            }
        )

    return Spine(
        simulation_state=simulation_state_df,
        scheduled_events=scheduled_events_df,
    )
