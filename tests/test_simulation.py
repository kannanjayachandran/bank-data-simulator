"""Tests for SimulationConfig parameters and validation."""

from datetime import date
import pytest
from pydantic import ValidationError
from config.simulation import SimulationConfig


def test_simulation_config_defaults():
    """Verify that the SimulationConfig default values are correct."""
    config = SimulationConfig()
    assert config.n_customers == 10000
    assert config.sim_start == date(2024, 1, 1)
    assert config.sim_months == 24
    assert config.seed == 42
    assert config.sim_end == date(2026, 1, 1)


def test_simulation_config_validation():
    """Verify field constraints and validations on SimulationConfig."""
    # Test valid configuration
    config = SimulationConfig(n_customers=5000, sim_months=12, seed=100)
    assert config.n_customers == 5000
    assert config.sim_months == 12
    assert config.seed == 100

    # Test invalid n_customers (negative or zero)
    with pytest.raises(ValidationError):
        SimulationConfig(n_customers=0)
    with pytest.raises(ValidationError):
        SimulationConfig(n_customers=-5)

    # Test invalid sim_months
    with pytest.raises(ValidationError):
        SimulationConfig(sim_months=0)
    with pytest.raises(ValidationError):
        SimulationConfig(sim_months=121)

    # Test invalid seed
    with pytest.raises(ValidationError):
        SimulationConfig(seed=-1)


def test_simulation_config_sim_end():
    """Verify that the sim_end derived property calculates the correct boundary date."""
    # Standard 24 month duration starting Jan 2024
    config = SimulationConfig(sim_start=date(2024, 1, 1), sim_months=24)
    assert config.sim_end == date(2026, 1, 1)

    # 12 month duration starting Dec 2024
    config = SimulationConfig(sim_start=date(2024, 12, 1), sim_months=12)
    assert config.sim_end == date(2025, 12, 1)

    # 1 month duration
    config = SimulationConfig(sim_start=date(2024, 6, 1), sim_months=1)
    assert config.sim_end == date(2024, 7, 1)

    # Cross-year boundary calculation
    config = SimulationConfig(sim_start=date(2024, 10, 1), sim_months=5)
    assert config.sim_end == date(2025, 3, 1)


def test_simulation_config_sim_start_normalization():
    """Verify that input start dates are normalized to the first day of the month."""
    config = SimulationConfig(sim_start=date(2024, 1, 15))
    assert config.sim_start == date(2024, 1, 1)
    assert config.sim_end == date(2026, 1, 1)

    config_end_of_month = SimulationConfig(sim_start=date(2024, 2, 29))
    assert config_end_of_month.sim_start == date(2024, 2, 1)

