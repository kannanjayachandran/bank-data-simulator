"""Tests for persona configuration correctness."""

from config.personas import Persona, PERSONA_CONFIGS, PersonaConfig


def test_persona_enum_members():
    """Verify that all expected personas are defined in the Persona enum."""
    expected_personas = {
        "salary_core",
        "affluent_multi_product",
        "digital_native",
        "credit_stressed",
        "dormant_wealthy",
        "complaint_prone_churner",
    }
    actual_personas = {p.value for p in Persona}
    assert actual_personas == expected_personas


def test_persona_configs_exist():
    """Verify that each Persona enum member has a corresponding configuration."""
    for persona in Persona:
        assert persona in PERSONA_CONFIGS
        config = PERSONA_CONFIGS[persona]
        assert isinstance(config, PersonaConfig)
        assert config.persona == persona


def test_persona_product_uptake_contains_all_11_products():
    """Verify that the product uptake dict on every persona contains all 11 products."""
    expected_products = {
        "savings",
        "current",
        "debit_card",
        "credit_card",
        "personal_loan",
        "home_loan",
        "fixed_deposit",
        "insurance",
        "mutual_fund",
        "demat_account",
        "wealth_management",
    }

    for persona, config in PERSONA_CONFIGS.items():
        actual_products = set(config.product_uptake_probs.keys())
        missing_products = expected_products - actual_products
        extra_products = actual_products - expected_products

        assert not missing_products, f"{persona.value} is missing products: {missing_products}"
        assert not extra_products, f"{persona.value} has extra unexpected products: {extra_products}"

        # Verify that all probabilities are between 0 and 1
        for product, prob in config.product_uptake_probs.items():
            assert 0.0 <= prob <= 1.0, f"{persona.value} has invalid probability for {product}: {prob}"


def test_persona_parameter_sanity_bounds():
    """Verify bounds and logic for persona configuration parameters."""
    for persona, config in PERSONA_CONFIGS.items():
        # Income clips logic
        assert 0.0 < config.income_clip_min < config.income_clip_max
        assert config.income_log_mu > 0.0
        assert config.income_log_sigma > 0.0

        # Beta distributions positive
        assert config.digital_engagement_beta_a > 0.0
        assert config.digital_engagement_beta_b > 0.0

        # Complaint rate logic
        assert 0.0 <= config.complaint_rate_min <= config.complaint_rate_max <= 1.0

        # Base monthly churn logic
        assert 0.0 <= config.base_monthly_churn_min <= config.base_monthly_churn_max <= 1.0

        # Sigma noise positive
        assert 0.0 < config.sigma_noise <= 0.50

        # Preferences range check [0, 1]
        assert 0.0 <= config.cash_pref <= 1.0
        assert 0.0 <= config.branch_pref <= 1.0
        assert 0.0 <= config.loan_propensity <= 1.0
        assert 0.0 <= config.low_sensitivity_share <= 1.0


def test_cross_persona_invariants():
    """Verify relative behavioral relationships between personas to catch copy-paste errors."""
    salary_core = PERSONA_CONFIGS[Persona.SALARY_CORE]
    affluent = PERSONA_CONFIGS[Persona.AFFLUENT_MULTI_PRODUCT]
    digital_native = PERSONA_CONFIGS[Persona.DIGITAL_NATIVE]
    credit_stressed = PERSONA_CONFIGS[Persona.CREDIT_STRESSED]
    dormant_wealthy = PERSONA_CONFIGS[Persona.DORMANT_WEALTHY]
    complaint_prone = PERSONA_CONFIGS[Persona.COMPLAINT_PRONE_CHURNER]

    # Churn ordering: credit_stressed and complaint_prone are high churn risk
    assert credit_stressed.base_monthly_churn_max > salary_core.base_monthly_churn_max
    assert complaint_prone.base_monthly_churn_max > affluent.base_monthly_churn_max

    # Complaint rate ordering: complaint_prone has the highest complaint propensity
    for p, config in PERSONA_CONFIGS.items():
        if p != Persona.COMPLAINT_PRONE_CHURNER:
            assert complaint_prone.complaint_rate_max > config.complaint_rate_max
            assert complaint_prone.complaint_rate_min > config.complaint_rate_min

    # Income ordering: affluent and dormant_wealthy are higher than core salary
    assert affluent.income_clip_min >= salary_core.income_clip_max
    assert dormant_wealthy.income_clip_min >= salary_core.income_clip_max

    # Digital engagement ordering: digital_native is highest, dormant_wealthy is lowest
    def beta_mean(a: float, b: float) -> float:
        return a / (a + b)

    dn_mean = beta_mean(digital_native.digital_engagement_beta_a, digital_native.digital_engagement_beta_b)
    dw_mean = beta_mean(dormant_wealthy.digital_engagement_beta_a, dormant_wealthy.digital_engagement_beta_b)
    
    assert dn_mean > dw_mean
    for p, config in PERSONA_CONFIGS.items():
        if p != Persona.DIGITAL_NATIVE:
            p_mean = beta_mean(config.digital_engagement_beta_a, config.digital_engagement_beta_b)
            assert dn_mean > p_mean
        if p != Persona.DORMANT_WEALTHY:
            p_mean = beta_mean(config.digital_engagement_beta_a, config.digital_engagement_beta_b)
            assert dw_mean < p_mean

