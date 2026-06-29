"""Tests for the spine generator module (generator/spine.py)."""

import polars as pl
from config.constants import CUSTOMER_ID_START, LOW_SENSITIVITY_SHARE_RANGE
from config.events import UNCONDITIONAL_EVENTS, HiddenEvent
from config.simulation import SimulationConfig
from generator.spine import generate_spine


def test_spine_generation_counts_and_types():
    """Verify customer counts, unique IDs, and column types in generated spine."""
    config = SimulationConfig(n_customers=120, sim_months=12, seed=123)
    spine = generate_spine(config)

    # Verify simulation_state structure
    state = spine.simulation_state
    assert len(state) == 120
    assert state.columns == [
        "customer_id",
        "persona",
        "low_sensitivity_segment",
        "churn_month",
        "churned_flag",
        "churn_reason",
        "active_months_generated",
    ]

    # Verify Customer IDs are unique and sequential
    assert state["customer_id"].is_unique().all()
    assert state["customer_id"].min() == CUSTOMER_ID_START
    assert state["customer_id"].max() == CUSTOMER_ID_START + 119

    # Verify column types
    assert state["customer_id"].dtype == pl.Int64
    assert state["persona"].dtype == pl.String
    assert state["low_sensitivity_segment"].dtype == pl.Boolean
    assert state["churn_month"].dtype == pl.Date
    assert state["churned_flag"].dtype == pl.Boolean
    assert state["churn_reason"].dtype == pl.String
    assert state["active_months_generated"].dtype == pl.Int32


def test_persona_distribution_and_low_sensitivity():
    """Verify that personas are uniformly distributed and low sensitivity flags are correctly sampled."""
    config = SimulationConfig(n_customers=600, sim_months=12, seed=42)
    spine = generate_spine(config)

    state = spine.simulation_state

    # Verify that all 6 personas are assigned and counts are balanced
    persona_counts = state["persona"].value_counts()
    assert len(persona_counts) == 6
    for count_row in persona_counts.iter_rows():
        # With 600 customers, uniform assignment should give roughly ~100 per persona.
        # We assert between 60 and 140 to allow for stochastic variation from rng.choice.
        assert 60 <= count_row[1] <= 140

    # Verify low_sensitivity_segment share falls within the expected range
    low_sens_count = state["low_sensitivity_segment"].sum()
    low_sens_share = low_sens_count / len(state)
    assert (
        LOW_SENSITIVITY_SHARE_RANGE[0]
        <= low_sens_share
        <= LOW_SENSITIVITY_SHARE_RANGE[1]
    )


def test_scheduled_events_structure_and_types():
    """Verify scheduled_events schema and column invariants."""
    config = SimulationConfig(n_customers=50, sim_months=24, seed=42)
    spine = generate_spine(config)

    events = spine.scheduled_events
    assert events.columns == ["customer_id", "event_month", "event_type", "is_fired"]

    # Verify types
    assert events["customer_id"].dtype == pl.Int64
    assert events["event_month"].dtype == pl.Date
    assert events["event_type"].dtype == pl.String
    assert events["is_fired"].dtype == pl.Boolean

    # All is_fired values must be initialized to False
    assert not events["is_fired"].any()

    # Only unconditional events must be scheduled
    unconditional_vals = {e.value for e in UNCONDITIONAL_EVENTS}
    scheduled_vals = set(events["event_type"].unique().to_list())
    assert scheduled_vals.issubset(unconditional_vals)


def test_event_scheduling_mutual_exclusions():
    """Verify that scheduled events satisfy mutual exclusion rules (salary, expense)."""
    config = SimulationConfig(n_customers=1000, sim_months=24, seed=999)
    spine = generate_spine(config)
    events = spine.scheduled_events

    # Group by customer_id and event_month to detect collisions
    collisions = (
        events.group_by(["customer_id", "event_month"])
        .agg(pl.col("event_type"))
        .filter(pl.col("event_type").list.len() > 1)
    )

    # Inspect overlapping lists
    for row in collisions.iter_rows():
        event_list = row[2]
        # Salary category rule: job change and delay are mutually exclusive
        assert not (
            HiddenEvent.SALARY_JOB_CHANGE.value in event_list
            and HiddenEvent.SALARY_DELAY.value in event_list
        )

        # Expense category rule: home purchase and large life expense are mutually exclusive
        assert not (
            HiddenEvent.HOME_PURCHASE.value in event_list
            and HiddenEvent.LARGE_LIFE_EXPENSE.value in event_list
        )


def test_spine_determinism_and_reproducibility():
    """Verify that identical seed yields identical results, and different seeds differ."""
    config1 = SimulationConfig(n_customers=100, sim_months=12, seed=42)
    config2 = SimulationConfig(n_customers=100, sim_months=12, seed=42)
    config3 = SimulationConfig(n_customers=100, sim_months=12, seed=99)

    spine1 = generate_spine(config1)
    spine2 = generate_spine(config2)
    spine3 = generate_spine(config3)

    # Identical seeds
    assert spine1.simulation_state.equals(spine2.simulation_state)
    assert spine1.scheduled_events.equals(spine2.scheduled_events)

    # Different seeds
    assert not spine1.simulation_state.equals(spine3.simulation_state)
