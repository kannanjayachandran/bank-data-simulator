"""Churn scoring model for the synthetic retail banking universe.

Defines the ChurnInput and ChurnResult dataclasses, and the calculate_churn function.
Evaluates hard triggers first, then falls back to a weighted risk scoring formula
with Gaussian noise and Bernoulli or thresholded churn status decisions.
"""

from dataclasses import dataclass
import math
from typing import Optional
import numpy as np

from config.personas import (
    CHURN_COMPONENT_WEIGHTS,
    CHURN_THRESHOLDS,
    PERSONA_CONFIGS,
    Persona,
)


@dataclass
class ChurnInput:
    """Input parameters representing a customer's behavioral and risk state.

    All fields are defaulted to allow early simulation month execution.
    """

    persona: Persona

    # Raw scoring component values (unweighted)
    base_rate: float = 0.0
    event_score: float = 0.0
    trend_score: float = 0.0
    product_score: float = 0.0
    complaint_score: float = 0.0
    loan_stress_score: float = 0.0
    digital_inactivity_score: float = 0.0

    # Hard triggers & priority checks
    dpd_days: int = 0
    loan_status: str = "Active"
    complaint_count_6m: int = 0
    unresolved_complaints: int = 0
    current_balance: float = 1000.0
    months_without_salary: int = 0
    service_failures_2m: int = 0
    digital_inactive_months: int = 0
    core_account_closed: bool = False

    # Churn reason mapping helpers
    recent_salary_job_change: bool = False
    products_count_drop: bool = False
    recent_service_failure: bool = False


@dataclass(frozen=True)
class ChurnResult:
    """Structure storing the outcome of the churn scoring evaluation."""

    churned: bool
    churn_risk: float
    churn_prob: float
    churn_reason: Optional[str] = None


def sigmoid(x: float) -> float:
    """Computes a numerically stable sigmoid function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def calculate_churn(
    input_data: ChurnInput, rng: np.random.Generator, use_threshold: bool = False
) -> ChurnResult:
    """Calculates churn risk, probability, outcome, and reason for a customer snapshot.

    Args:
        input_data: Current month metrics for a customer.
        rng: Seeded numpy random generator (passed from simulation loop).
        use_threshold: If True, evaluates churn status deterministically against persona threshold.
                       If False (default), evaluates churn status via Bernoulli trial.

    Returns:
        ChurnResult: Churn decision, risk score, probability, and reason.
    """
    # 1. Evaluate Churn Triggers (Hard Triggers & Salary Loss)
    # These force churn (prob=1.0) and map directly to specific reasons.
    is_loan_default = (
        input_data.loan_status == "Delinquent" and input_data.dpd_days >= 90
    )
    is_salary_lost = (
        input_data.recent_salary_job_change and input_data.months_without_salary >= 2
    )
    is_service_dissatisfaction = (
        input_data.complaint_count_6m >= 4 and input_data.unresolved_complaints >= 2
    )
    is_dormancy = (
        input_data.current_balance < 500.0 and input_data.months_without_salary >= 3
    )
    is_service_failure = (
        input_data.service_failures_2m >= 2 and input_data.digital_inactive_months >= 2
    )
    is_voluntary_close = input_data.core_account_closed

    has_hard_trigger = (
        is_loan_default
        or is_salary_lost
        or is_service_dissatisfaction
        or is_dormancy
        or is_service_failure
        or is_voluntary_close
    )

    if has_hard_trigger:
        # Map reason based on priority rules (spec Section 10)
        if is_loan_default:
            reason = "Loan default"
        elif is_salary_lost:
            reason = "Salary account lost"
        elif is_service_dissatisfaction:
            reason = "Service dissatisfaction"
        elif is_dormancy:
            reason = "Account dormancy"
        elif is_service_failure:
            reason = "Service failure"
        else:
            reason = "Voluntary closure"

        return ChurnResult(
            churned=True,
            churn_risk=999.0,  # High risk value for hard triggers
            churn_prob=1.0,
            churn_reason=reason,
        )

    # 2. Score calculation using component weights and Gaussian noise (spec Section 9.1 & 9.2)
    p_config = PERSONA_CONFIGS[input_data.persona]
    noise = rng.normal(0, p_config.sigma_noise)

    w = CHURN_COMPONENT_WEIGHTS
    churn_risk = (
        input_data.base_rate
        + input_data.event_score * w["event_score"]
        + input_data.trend_score * w["trend_score"]
        + input_data.product_score * w["product_score"]
        + input_data.complaint_score * w["complaint_score"]
        + input_data.loan_stress_score * w["loan_stress_score"]
        + input_data.digital_inactivity_score * w["digital_inactivity_score"]
        + noise
    )

    churn_prob = sigmoid(churn_risk)

    # 3. Churn Status Decision (Bernoulli or Persona Threshold check)
    if use_threshold:
        threshold = CHURN_THRESHOLDS[input_data.persona]
        churned = churn_prob > threshold
    else:
        churned = rng.random() < churn_prob

    # 4. Reason Assignment (spec Section 10 Priority Rules)
    churn_reason = None
    if churned:
        if input_data.dpd_days >= 90 or input_data.loan_status == "Delinquent":
            churn_reason = "Loan default"
        elif (
            input_data.months_without_salary >= 2
            and input_data.recent_salary_job_change
        ):
            churn_reason = "Salary account lost"
        elif (
            input_data.unresolved_complaints >= 1 or input_data.complaint_count_6m >= 2
        ):
            churn_reason = "Service dissatisfaction"
        elif (
            input_data.current_balance < 500.0
            or input_data.digital_inactive_months >= 2
        ):
            churn_reason = "Account dormancy"
        elif input_data.products_count_drop:
            churn_reason = "Product disengagement"
        elif input_data.recent_service_failure:
            churn_reason = "Service failure"
        else:
            # Fallback to Voluntary closure as catch-all
            churn_reason = "Voluntary closure"

    return ChurnResult(
        churned=churned,
        churn_risk=churn_risk,
        churn_prob=churn_prob,
        churn_reason=churn_reason,
    )
