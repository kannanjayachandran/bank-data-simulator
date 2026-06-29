"""Simulation configuration and parameter validation.

Defines the Pydantic-based SimulationConfig class to represent the top-level
parameters of the synthetic data generation run.
"""

from datetime import date

from pydantic import BaseModel, Field, field_validator


class SimulationConfig(BaseModel):
    """Configuration class for the simulation execution parameters.

    Uses Pydantic for automated field validation.
    """

    # Note: Development default is set to 2000 customers.
    # The production target is 100,000 customers or beyond.
    n_customers: int = Field(
        default=2000,
        description="Total number of customers to generate in the synthetic spine.",
    )

    sim_start: date = Field(
        default=date(2024, 1, 1),
        description="The start date of the simulation. Always normalized to the first day of the month.",
    )

    sim_months: int = Field(
        default=24,
        description="The duration of the simulation run in months.",
    )

    seed: int = Field(
        default=42,
        description="Global seed for reproducibility across randomized components.",
    )

    @field_validator("n_customers")
    @classmethod
    def validate_n_customers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_customers must be strictly positive (> 0)")
        return v

    @field_validator("sim_start")
    @classmethod
    def normalize_sim_start(cls, v: date) -> date:
        """Ensure sim_start is always normalized to the first day of the month."""
        return date(v.year, v.month, 1)

    @field_validator("sim_months")
    @classmethod
    def validate_sim_months(cls, v: int) -> int:
        if not (1 <= v <= 120):
            raise ValueError(
                "sim_months must be in the range [1, 120] (up to 10 years)"
            )
        return v

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be non-negative (>= 0)")
        return v

    @property
    def sim_end(self) -> date:
        """Derived property representing the exclusive boundary date after the simulation ends.

        Calculated as sim_start + sim_months.
        For example: date(2024, 1, 1) + 24 months -> date(2026, 1, 1).
        """
        # Calculate new month and year manually to avoid dependencies
        start_month_index = self.sim_start.month - 1
        end_month_index = start_month_index + self.sim_months

        new_year = self.sim_start.year + (end_month_index // 12)
        new_month = (end_month_index % 12) + 1

        return date(new_year, new_month, 1)
