"""Tests for hidden event configuration correctness."""

from config.events import (
    HiddenEvent,
    EVENT_PROBABILITIES,
    UNCONDITIONAL_EVENTS,
    CONDITIONAL_EVENTS,
    CONDITIONAL_EVENT_BASELINES,
)
from config.personas import Persona


def test_event_enum_members():
    """Verify that all expected events are defined in the HiddenEvent enum."""
    expected_events = {
        "salary_job_change",
        "salary_delay",
        "large_life_expense",
        "home_purchase",
        "marriage_or_family_change",
        "relocation",
        "bank_service_failure",
        "card_decline_spike",
        "fee_hike_or_service_charge",
        "campaign_exposure",
        "loan_delinquency_start",
        "complaint_resolved",
    }
    actual_events = {e.value for e in HiddenEvent}
    assert actual_events == expected_events


def test_conditional_events_invariant():
    """Verify that conditional events are strictly absent from the static EVENT_PROBABILITIES table."""
    for persona in Persona:
        mapped_events = set(EVENT_PROBABILITIES[persona].keys())
        overlap = mapped_events & CONDITIONAL_EVENTS
        assert not overlap, (
            f"Conditional event(s) {overlap} present in EVENT_PROBABILITIES for persona {persona.value}! "
            "Conditional events must only resolve dynamically at simulation time and be absent from this table."
        )


def test_event_probabilities_mapping_completeness():
    """Verify that EVENT_PROBABILITIES contains all unconditional events for all personas."""
    for persona in Persona:
        assert persona in EVENT_PROBABILITIES
        mapped_events = set(EVENT_PROBABILITIES[persona].keys())

        # Check for missing unconditional events
        missing_unconditional = UNCONDITIONAL_EVENTS - mapped_events
        assert not missing_unconditional, (
            f"Persona {persona.value} is missing unconditional event configurations: {missing_unconditional}"
        )

        # Check for extra unexpected events
        unexpected_events = mapped_events - UNCONDITIONAL_EVENTS
        assert not unexpected_events, (
            f"Persona {persona.value} has unexpected events configured in EVENT_PROBABILITIES: {unexpected_events}"
        )

        # Verify probability bounds [0, 1]
        for event, prob in EVENT_PROBABILITIES[persona].items():
            assert 0.0 <= prob <= 1.0, (
                f"Invalid probability for {event.value} in persona {persona.value}: {prob}"
            )


def test_conditional_event_baselines_completeness():
    """Verify that CONDITIONAL_EVENT_BASELINES contains all conditional events for all personas."""
    for persona in Persona:
        assert persona in CONDITIONAL_EVENT_BASELINES
        mapped_events = set(CONDITIONAL_EVENT_BASELINES[persona].keys())

        # Check for missing conditional events
        missing_conditional = CONDITIONAL_EVENTS - mapped_events
        assert not missing_conditional, (
            f"Persona {persona.value} is missing conditional event configurations: {missing_conditional}"
        )

        # Verify probability bounds [0, 1]
        for event, prob in CONDITIONAL_EVENT_BASELINES[persona].items():
            assert 0.0 <= prob <= 1.0, (
                f"Invalid baseline probability for {event.value} in persona {persona.value}: {prob}"
            )
