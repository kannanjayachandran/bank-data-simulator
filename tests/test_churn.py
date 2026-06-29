"""Tests for the churn scoring module (generator/churn.py)."""

import numpy as np
from config.personas import Persona
from generator.churn import ChurnInput, calculate_churn


def test_safe_sigmoid_limits():
    """Verify that sigmoid calculations are stable for extreme risk scores."""
    from generator.churn import sigmoid

    assert math_close(sigmoid(0), 0.5)
    assert math_close(sigmoid(100), 1.0)
    assert math_close(sigmoid(-100), 0.0)


def math_close(a: float, b: float, tol: float = 1e-5) -> bool:
    return abs(a - b) <= tol


def test_normal_scoring_weights_and_noise():
    """Verify weighted risk score calculation and Gaussian noise integration."""
    rng = np.random.default_rng(123)

    # Base input with 0 for all scores
    inp = ChurnInput(
        persona=Persona.SALARY_CORE,
        base_rate=0.1,
    )
    result = calculate_churn(inp, rng, use_threshold=True)

    # Churn risk should be base_rate (0.1) + weighted scores (0.0) + noise.
    # Noise for SALARY_CORE (sigma_noise=0.06) drawn with seed 123 -> approx -0.059
    assert -0.2 < result.churn_risk < 0.4
    assert 0.0 < result.churn_prob < 1.0
    assert not result.churned

    # Verify component scores alter risk in expected direction
    inp_high_risk = ChurnInput(
        persona=Persona.SALARY_CORE,
        base_rate=0.1,
        complaint_score=2.0,  # weight = 0.25 -> risk increases by 0.5
        product_score=1.0,  # weight = -0.10 -> risk decreases by 0.1
    )
    rng2 = np.random.default_rng(123)  # same seed to align noise
    result_high = calculate_churn(inp_high_risk, rng2, use_threshold=True)

    expected_risk_diff = (2.0 * 0.25) + (1.0 * -0.10)  # 0.4
    assert math_close(result_high.churn_risk - result.churn_risk, expected_risk_diff)
    assert result_high.churn_prob > result.churn_prob


def test_hard_triggers_override_scoring():
    """Verify that hard triggers force immediate churn (prob=1.0) and map to correct reasons."""
    rng = np.random.default_rng(42)

    # 1. Delinquency (Loan default)
    inp_loan = ChurnInput(
        persona=Persona.SALARY_CORE,
        loan_status="Delinquent",
        dpd_days=95,
        base_rate=-5.0,  # naturally would never churn
    )
    res_loan = calculate_churn(inp_loan, rng)
    assert res_loan.churned
    assert res_loan.churn_prob == 1.0
    assert res_loan.churn_reason == "Loan default"

    # 2. Salary lost (Salary account lost)
    inp_salary = ChurnInput(
        persona=Persona.SALARY_CORE,
        recent_salary_job_change=True,
        months_without_salary=2,
        base_rate=-5.0,
    )
    res_salary = calculate_churn(inp_salary, rng)
    assert res_salary.churned
    assert res_salary.churn_prob == 1.0
    assert res_salary.churn_reason == "Salary account lost"

    # 3. Service dissatisfaction
    inp_service = ChurnInput(
        persona=Persona.SALARY_CORE,
        complaint_count_6m=4,
        unresolved_complaints=2,
        base_rate=-5.0,
    )
    res_service = calculate_churn(inp_service, rng)
    assert res_service.churned
    assert res_service.churn_prob == 1.0
    assert res_service.churn_reason == "Service dissatisfaction"

    # 4. Account dormancy
    inp_dormant = ChurnInput(
        persona=Persona.SALARY_CORE,
        current_balance=450.0,
        months_without_salary=3,
        base_rate=-5.0,
    )
    res_dormant = calculate_churn(inp_dormant, rng)
    assert res_dormant.churned
    assert res_dormant.churn_prob == 1.0
    assert res_dormant.churn_reason == "Account dormancy"

    # 5. Service failure
    inp_fail = ChurnInput(
        persona=Persona.SALARY_CORE,
        service_failures_2m=2,
        digital_inactive_months=2,
        base_rate=-5.0,
    )
    res_fail = calculate_churn(inp_fail, rng)
    assert res_fail.churned
    assert res_fail.churn_prob == 1.0
    assert res_fail.churn_reason == "Service failure"

    # 6. Voluntary closure
    inp_close = ChurnInput(
        persona=Persona.SALARY_CORE,
        core_account_closed=True,
        base_rate=-5.0,
    )
    res_close = calculate_churn(inp_close, rng)
    assert res_close.churned
    assert res_close.churn_prob == 1.0
    assert res_close.churn_reason == "Voluntary closure"


def test_hard_trigger_priority_mapping():
    """Verify priority ordering rule when multiple hard triggers trigger simultaneously."""
    rng = np.random.default_rng(42)

    # Loan default (priority 1) vs. Voluntary closure (priority 6)
    inp = ChurnInput(
        persona=Persona.SALARY_CORE,
        loan_status="Delinquent",
        dpd_days=95,
        core_account_closed=True,
    )
    res = calculate_churn(inp, rng)
    assert res.churn_reason == "Loan default"

    # Salary account lost (priority 2) vs. Account dormancy (priority 4)
    inp2 = ChurnInput(
        persona=Persona.SALARY_CORE,
        recent_salary_job_change=True,
        months_without_salary=3,  # satisfies salary lost (>=2) and dormancy (>=3 and balance < 500)
        current_balance=100.0,
    )
    res2 = calculate_churn(inp2, rng)
    assert res2.churn_reason == "Salary account lost"


def test_normal_churn_reason_priorities():
    """Verify priority mapping rules for standard risk-based churn outcomes."""
    rng = np.random.default_rng(42)

    # Setup high risk to force churn
    base_inp = ChurnInput(
        persona=Persona.SALARY_CORE,
        base_rate=10.0,  # forces churn_prob approx 1.0
    )

    # Loan default condition met
    inp_loan = base_inp
    inp_loan.loan_status = "Delinquent"
    res_loan = calculate_churn(inp_loan, rng, use_threshold=True)
    assert res_loan.churn_reason == "Loan default"

    # Salary account lost condition met (but not loan)
    inp_sal = ChurnInput(
        persona=Persona.SALARY_CORE,
        base_rate=10.0,
        recent_salary_job_change=True,
        months_without_salary=2,
    )
    res_sal = calculate_churn(inp_sal, rng, use_threshold=True)
    assert res_sal.churn_reason == "Salary account lost"

    # Product disengagement met (and not higher priorities)
    inp_prod = ChurnInput(
        persona=Persona.SALARY_CORE,
        base_rate=10.0,
        products_count_drop=True,
    )
    res_prod = calculate_churn(inp_prod, rng, use_threshold=True)
    assert res_prod.churn_reason == "Product disengagement"

    # Service failure met (and not higher priorities)
    inp_serv = ChurnInput(
        persona=Persona.SALARY_CORE,
        base_rate=10.0,
        recent_service_failure=True,
    )
    res_serv = calculate_churn(inp_serv, rng, use_threshold=True)
    assert res_serv.churn_reason == "Service failure"

    # Fallback default
    inp_fallback = ChurnInput(persona=Persona.SALARY_CORE, base_rate=10.0)
    res_fallback = calculate_churn(inp_fallback, rng, use_threshold=True)
    assert res_fallback.churn_reason == "Voluntary closure"


def test_threshold_vs_bernoulli_churn():
    """Verify behavior differences between thresholded (deterministic) and Bernoulli (probabilistic) churn decisions."""
    # Verify that Bernoulli decision behaves statistically
    # over many trials.
    bernoulli_churns = 0
    inp_mid = ChurnInput(
        persona=Persona.SALARY_CORE, base_rate=0.0
    )  # churn_prob will be around 0.5
    for seed in range(500):
        rng_t = np.random.default_rng(seed)
        res = calculate_churn(inp_mid, rng_t, use_threshold=False)
        if res.churned:
            bernoulli_churns += 1

    # With prob ~0.5 over 500 trials, churns should be roughly centered
    assert 150 < bernoulli_churns < 350
