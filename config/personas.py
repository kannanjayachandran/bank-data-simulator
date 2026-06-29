"""Persona configurations

Defines the Persona enum, the PersonaConfig dataclass, and the static configuration
parameters for the six primary target personas.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Persona(str, Enum):
    SALARY_CORE = "salary_core"
    AFFLUENT_MULTI_PRODUCT = "affluent_multi_product"
    DIGITAL_NATIVE = "digital_native"
    CREDIT_STRESSED = "credit_stressed"
    DORMANT_WEALTHY = "dormant_wealthy"
    COMPLAINT_PRONE_CHURNER = "complaint_prone_churner"


@dataclass(frozen=True)
class PersonaConfig:
    """Configuration class for persona-specific statistical attributes."""

    persona: Persona

    # Lognormal income parameters
    income_log_mu: float
    income_log_sigma: float
    income_clip_min: float
    income_clip_max: float

    # Beta distribution parameters for digital engagement [0, 1]
    digital_engagement_beta_a: float
    digital_engagement_beta_b: float

    # Complaint rate boundaries (monthly complaint probability)
    complaint_rate_min: float
    complaint_rate_max: float

    # Base monthly churn probabilities
    base_monthly_churn_min: float
    base_monthly_churn_max: float

    # Probability of holding a product (used in portfolio generation / cross-selling)
    # Must contain exactly 11 products:
    # savings, current, debit_card, credit_card, personal_loan,
    # home_loan, fixed_deposit, insurance, mutual_fund, demat_account, wealth_management
    product_uptake_probs: Dict[str, float] = field(default_factory=dict)

    # General preference and propensity scalars
    loan_propensity: float = 0.05
    cash_pref: float = 0.15
    branch_pref: float = 0.10

    # Share of customers within this persona belonging to the low-sensitivity segment
    low_sensitivity_share: float = 0.12

    # Churn Scorer Noise (sigma_persona)
    sigma_noise: float = 0.08


# Static Configurations for all 6 personas
PERSONA_CONFIGS: Dict[Persona, PersonaConfig] = {
    Persona.SALARY_CORE: PersonaConfig(
        persona=Persona.SALARY_CORE,
        # ₹3L–₹12L
        income_log_mu=13.30,
        income_log_sigma=0.35,
        income_clip_min=300000.0,
        income_clip_max=1200000.0,
        # high digital engagement
        digital_engagement_beta_a=7.0,
        digital_engagement_beta_b=3.0,
        # low complaint rate
        complaint_rate_min=0.005,
        complaint_rate_max=0.02,
        # 0.3%–1.0% base monthly churn
        base_monthly_churn_min=0.003,
        base_monthly_churn_max=0.010,
        product_uptake_probs={
            "savings": 0.98,
            "current": 0.08,
            "debit_card": 0.95,
            "credit_card": 0.35,
            "personal_loan": 0.10,
            "home_loan": 0.08,
            "fixed_deposit": 0.22,
            "insurance": 0.20,
            "mutual_fund": 0.12,
            "demat_account": 0.08,
            "wealth_management": 0.01,
        },
        loan_propensity=0.02,
        cash_pref=0.30,
        branch_pref=0.20,
        low_sensitivity_share=0.12,
        sigma_noise=0.06,  # Low-risk persona noise
    ),
    Persona.AFFLUENT_MULTI_PRODUCT: PersonaConfig(
        persona=Persona.AFFLUENT_MULTI_PRODUCT,
        # ₹12L–₹60L
        income_log_mu=14.80,
        income_log_sigma=0.40,
        income_clip_min=1200000.0,
        income_clip_max=6000000.0,
        # medium-high digital engagement
        digital_engagement_beta_a=6.0,
        digital_engagement_beta_b=4.0,
        # low complaint rate
        complaint_rate_min=0.005,
        complaint_rate_max=0.02,
        # 0.1%–0.4% base monthly churn
        base_monthly_churn_min=0.001,
        base_monthly_churn_max=0.004,
        product_uptake_probs={
            "savings": 0.97,
            "current": 0.20,
            "debit_card": 0.90,
            "credit_card": 0.55,
            "personal_loan": 0.08,
            "home_loan": 0.30,
            "fixed_deposit": 0.80,
            "insurance": 0.75,
            "mutual_fund": 0.78,
            "demat_account": 0.35,
            "wealth_management": 0.20,
        },
        loan_propensity=0.05,
        cash_pref=0.05,
        branch_pref=0.15,
        low_sensitivity_share=0.15,
        sigma_noise=0.05,  # Lowest-risk persona noise
    ),
    Persona.DIGITAL_NATIVE: PersonaConfig(
        persona=Persona.DIGITAL_NATIVE,
        # ₹4L–₹20L
        income_log_mu=13.70,
        income_log_sigma=0.40,
        income_clip_min=400000.0,
        income_clip_max=2000000.0,
        # very high digital engagement
        digital_engagement_beta_a=9.0,
        digital_engagement_beta_b=1.0,
        # low-medium complaint rate
        complaint_rate_min=0.01,
        complaint_rate_max=0.04,
        # 0.1%–0.5% base monthly churn
        base_monthly_churn_min=0.001,
        base_monthly_churn_max=0.005,
        product_uptake_probs={
            "savings": 0.96,
            "current": 0.10,
            "debit_card": 0.92,
            "credit_card": 0.42,
            "personal_loan": 0.12,
            "home_loan": 0.04,
            "fixed_deposit": 0.10,
            "insurance": 0.12,
            "mutual_fund": 0.18,
            "demat_account": 0.20,
            "wealth_management": 0.01,
        },
        loan_propensity=0.03,
        cash_pref=0.02,
        branch_pref=0.01,
        low_sensitivity_share=0.10,
        sigma_noise=0.06,  # Low-risk persona noise
    ),
    Persona.CREDIT_STRESSED: PersonaConfig(
        persona=Persona.CREDIT_STRESSED,
        # ₹2.5L–₹10L
        income_log_mu=13.12,
        income_log_sigma=0.35,
        income_clip_min=250000.0,
        income_clip_max=1000000.0,
        # medium digital engagement
        digital_engagement_beta_a=5.0,
        digital_engagement_beta_b=5.0,
        # high complaint rate
        complaint_rate_min=0.05,
        complaint_rate_max=0.15,
        # 0.5%–1.5% base monthly churn
        base_monthly_churn_min=0.005,
        base_monthly_churn_max=0.015,
        product_uptake_probs={
            "savings": 0.94,
            "current": 0.05,
            "debit_card": 0.85,
            "credit_card": 0.65,
            "personal_loan": 0.45,
            "home_loan": 0.10,
            "fixed_deposit": 0.12,
            "insurance": 0.08,
            "mutual_fund": 0.06,
            "demat_account": 0.05,
            "wealth_management": 0.01,
        },
        loan_propensity=0.15,
        cash_pref=0.20,
        branch_pref=0.10,
        low_sensitivity_share=0.12,
        sigma_noise=0.10,  # High-risk persona noise
    ),
    Persona.DORMANT_WEALTHY: PersonaConfig(
        persona=Persona.DORMANT_WEALTHY,
        # ₹15L–₹1Cr
        income_log_mu=15.17,
        income_log_sigma=0.45,
        income_clip_min=1500000.0,
        income_clip_max=10000000.0,
        # low digital engagement
        digital_engagement_beta_a=1.5,
        digital_engagement_beta_b=8.5,
        # low complaint rate
        complaint_rate_min=0.002,
        complaint_rate_max=0.01,
        # 0.4%–1.5% base monthly churn
        base_monthly_churn_min=0.004,
        base_monthly_churn_max=0.015,
        product_uptake_probs={
            "savings": 0.95,
            "current": 0.12,
            "debit_card": 0.80,
            "credit_card": 0.25,
            "personal_loan": 0.03,
            "home_loan": 0.18,
            "fixed_deposit": 0.88,
            "insurance": 0.70,
            "mutual_fund": 0.72,
            "demat_account": 0.18,
            "wealth_management": 0.30,
        },
        loan_propensity=0.01,
        cash_pref=0.10,
        branch_pref=0.30,
        low_sensitivity_share=0.12,
        sigma_noise=0.08,  # Medium-risk noise
    ),
    Persona.COMPLAINT_PRONE_CHURNER: PersonaConfig(
        persona=Persona.COMPLAINT_PRONE_CHURNER,
        # ₹3L–₹15L
        income_log_mu=13.42,
        income_log_sigma=0.40,
        income_clip_min=300000.0,
        income_clip_max=1500000.0,
        # medium digital engagement
        digital_engagement_beta_a=5.0,
        digital_engagement_beta_b=5.0,
        # very high complaint rate
        complaint_rate_min=0.10,
        complaint_rate_max=0.30,
        # 1.0%–2.5% base monthly churn
        base_monthly_churn_min=0.010,
        base_monthly_churn_max=0.025,
        product_uptake_probs={
            "savings": 0.93,
            "current": 0.10,
            "debit_card": 0.78,
            "credit_card": 0.45,
            "personal_loan": 0.20,
            "home_loan": 0.05,
            "fixed_deposit": 0.15,
            "insurance": 0.10,
            "mutual_fund": 0.08,
            "demat_account": 0.06,
            "wealth_management": 0.01,
        },
        loan_propensity=0.05,
        cash_pref=0.15,
        branch_pref=0.25,
        low_sensitivity_share=0.10,
        sigma_noise=0.12,  # Highest-risk persona noise
    ),
}

# Churn thresholds by persona
CHURN_THRESHOLDS: Dict[Persona, float] = {
    Persona.SALARY_CORE: 0.72,
    Persona.AFFLUENT_MULTI_PRODUCT: 0.78,
    Persona.DIGITAL_NATIVE: 0.70,
    Persona.CREDIT_STRESSED: 0.58,
    Persona.DORMANT_WEALTHY: 0.74,
    Persona.COMPLAINT_PRONE_CHURNER: 0.55,
}

# Explicit component weights for Churn Scorer risk scoring formula
# Used to scale the raw component values during churn scoring calculation
CHURN_COMPONENT_WEIGHTS: Dict[str, float] = {
    "event_score": 0.30,
    "trend_score": 0.15,
    "product_score": -0.10,  # Negative, representing risk mitigation of holding multiple products
    "complaint_score": 0.25,
    "loan_stress_score": 0.20,
    "digital_inactivity_score": 0.15,
}
