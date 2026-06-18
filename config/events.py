"""Hidden event configuration

Defines the HiddenEvent enum and the probability tables for event triggers
based on customer personas.
"""

from enum import Enum
from typing import Dict, Set
from config.personas import Persona


class HiddenEvent(str, Enum):
    # Unconditional Events (can be pre-assigned or scheduled directly)
    SALARY_JOB_CHANGE = "salary_job_change"
    SALARY_DELAY = "salary_delay"
    LARGE_LIFE_EXPENSE = "large_life_expense"
    HOME_PURCHASE = "home_purchase"
    MARRIAGE_OR_FAMILY_CHANGE = "marriage_or_family_change"
    RELOCATION = "relocation"
    BANK_SERVICE_FAILURE = "bank_service_failure"
    FEE_HIKE_OR_SERVICE_CHARGE = "fee_hike_or_service_charge"

    # Conditional Events (triggered dynamically at simulation time based on prerequisites)
    CARD_DECLINE_SPIKE = "card_decline_spike"
    CAMPAIGN_EXPOSURE = "campaign_exposure"
    LOAN_DELINQUENCY_START = "loan_delinquency_start"
    COMPLAINT_RESOLVED = "complaint_resolved"


# Explicit sets of event types for validation
UNCONDITIONAL_EVENTS: Set[HiddenEvent] = {
    HiddenEvent.SALARY_JOB_CHANGE,
    HiddenEvent.SALARY_DELAY,
    HiddenEvent.LARGE_LIFE_EXPENSE,
    HiddenEvent.HOME_PURCHASE,
    HiddenEvent.MARRIAGE_OR_FAMILY_CHANGE,
    HiddenEvent.RELOCATION,
    HiddenEvent.BANK_SERVICE_FAILURE,
    HiddenEvent.FEE_HIKE_OR_SERVICE_CHARGE,
}

CONDITIONAL_EVENTS: Set[HiddenEvent] = {
    HiddenEvent.CARD_DECLINE_SPIKE,
    HiddenEvent.CAMPAIGN_EXPOSURE,
    HiddenEvent.LOAN_DELINQUENCY_START,
    HiddenEvent.COMPLAINT_RESOLVED,
}

# EVENT_PROBABILITIES: Static monthly trigger probabilities for UNCONDITIONAL events by persona.
# Note: As per invariants, conditional events must be absent from this table.
EVENT_PROBABILITIES: Dict[Persona, Dict[HiddenEvent, float]] = {
    Persona.SALARY_CORE: {
        HiddenEvent.SALARY_JOB_CHANGE: 0.015,
        HiddenEvent.SALARY_DELAY: 0.030,
        HiddenEvent.LARGE_LIFE_EXPENSE: 0.020,
        HiddenEvent.HOME_PURCHASE: 0.004,
        HiddenEvent.MARRIAGE_OR_FAMILY_CHANGE: 0.008,
        HiddenEvent.RELOCATION: 0.006,
        HiddenEvent.BANK_SERVICE_FAILURE: 0.020,
        HiddenEvent.FEE_HIKE_OR_SERVICE_CHARGE: 0.020,
    },
    Persona.AFFLUENT_MULTI_PRODUCT: {
        HiddenEvent.SALARY_JOB_CHANGE: 0.010,
        HiddenEvent.SALARY_DELAY: 0.020,
        HiddenEvent.LARGE_LIFE_EXPENSE: 0.040,
        HiddenEvent.HOME_PURCHASE: 0.012,
        HiddenEvent.MARRIAGE_OR_FAMILY_CHANGE: 0.010,
        HiddenEvent.RELOCATION: 0.007,
        HiddenEvent.BANK_SERVICE_FAILURE: 0.020,
        HiddenEvent.FEE_HIKE_OR_SERVICE_CHARGE: 0.010,
    },
    Persona.DIGITAL_NATIVE: {
        HiddenEvent.SALARY_JOB_CHANGE: 0.010,
        HiddenEvent.SALARY_DELAY: 0.020,
        HiddenEvent.LARGE_LIFE_EXPENSE: 0.020,
        HiddenEvent.HOME_PURCHASE: 0.003,
        HiddenEvent.MARRIAGE_OR_FAMILY_CHANGE: 0.007,
        HiddenEvent.RELOCATION: 0.008,
        HiddenEvent.BANK_SERVICE_FAILURE: 0.030,
        HiddenEvent.FEE_HIKE_OR_SERVICE_CHARGE: 0.020,
    },
    Persona.CREDIT_STRESSED: {
        HiddenEvent.SALARY_JOB_CHANGE: 0.020,
        HiddenEvent.SALARY_DELAY: 0.040,
        HiddenEvent.LARGE_LIFE_EXPENSE: 0.050,
        HiddenEvent.HOME_PURCHASE: 0.006,
        HiddenEvent.MARRIAGE_OR_FAMILY_CHANGE: 0.008,
        HiddenEvent.RELOCATION: 0.010,
        HiddenEvent.BANK_SERVICE_FAILURE: 0.030,
        HiddenEvent.FEE_HIKE_OR_SERVICE_CHARGE: 0.040,
    },
    Persona.DORMANT_WEALTHY: {
        HiddenEvent.SALARY_JOB_CHANGE: 0.005,
        HiddenEvent.SALARY_DELAY: 0.010,
        HiddenEvent.LARGE_LIFE_EXPENSE: 0.020,
        HiddenEvent.HOME_PURCHASE: 0.008,
        HiddenEvent.MARRIAGE_OR_FAMILY_CHANGE: 0.006,
        HiddenEvent.RELOCATION: 0.010,
        HiddenEvent.BANK_SERVICE_FAILURE: 0.010,
        HiddenEvent.FEE_HIKE_OR_SERVICE_CHARGE: 0.010,
    },
    Persona.COMPLAINT_PRONE_CHURNER: {
        HiddenEvent.SALARY_JOB_CHANGE: 0.015,
        HiddenEvent.SALARY_DELAY: 0.030,
        HiddenEvent.LARGE_LIFE_EXPENSE: 0.030,
        HiddenEvent.HOME_PURCHASE: 0.002,
        HiddenEvent.MARRIAGE_OR_FAMILY_CHANGE: 0.007,
        HiddenEvent.RELOCATION: 0.008,
        HiddenEvent.BANK_SERVICE_FAILURE: 0.040,
        HiddenEvent.FEE_HIKE_OR_SERVICE_CHARGE: 0.040,
    },
}

# CONDITIONAL_EVENT_BASELINES usage contract:
#
# These values specify the baseline monthly probability of a conditional event triggering,
# provided that the customer satisfies the required prerequisite(s) during that simulation month.
#
# Event-specific Prerequisite Rules (Contract for Phase 2+):
# 1. CARD_DECLINE_SPIKE:
#    - Prerequisite: Customer holds at least one active card.
#    - Logic: If met, sample probability from this table. If not, probability is 0.0.
# 2. CAMPAIGN_EXPOSURE:
#    - Prerequisite: Customer has notification_opt_in is True and is digitally active.
#    - Logic: If met, sample probability from this table. If not, probability is 0.0.
# 3. LOAN_DELINQUENCY_START:
#    - Prerequisite: Customer has at least one active, non-delinquent loan.
#    - Logic: If met, sample probability from this table. If not, probability is 0.0.
# 4. COMPLAINT_RESOLVED:
#    - Prerequisite: Customer has at least one unresolved complaint.
#    - Logic: If met, sample probability from this table. If not, probability is 0.0.
#
# Kept separate from static EVENT_PROBABILITIES to satisfy test invariants and prevent leakages.
CONDITIONAL_EVENT_BASELINES: Dict[Persona, Dict[HiddenEvent, float]] = {
    Persona.SALARY_CORE: {
        HiddenEvent.CARD_DECLINE_SPIKE: 0.010,
        HiddenEvent.CAMPAIGN_EXPOSURE: 0.100,
        HiddenEvent.LOAN_DELINQUENCY_START: 0.005,
        HiddenEvent.COMPLAINT_RESOLVED: 0.750,
    },
    Persona.AFFLUENT_MULTI_PRODUCT: {
        HiddenEvent.CARD_DECLINE_SPIKE: 0.010,
        HiddenEvent.CAMPAIGN_EXPOSURE: 0.150,
        HiddenEvent.LOAN_DELINQUENCY_START: 0.003,
        HiddenEvent.COMPLAINT_RESOLVED: 0.850,
    },
    Persona.DIGITAL_NATIVE: {
        HiddenEvent.CARD_DECLINE_SPIKE: 0.015,
        HiddenEvent.CAMPAIGN_EXPOSURE: 0.120,
        HiddenEvent.LOAN_DELINQUENCY_START: 0.005,
        HiddenEvent.COMPLAINT_RESOLVED: 0.800,
    },
    Persona.CREDIT_STRESSED: {
        HiddenEvent.CARD_DECLINE_SPIKE: 0.030,
        HiddenEvent.CAMPAIGN_EXPOSURE: 0.080,
        HiddenEvent.LOAN_DELINQUENCY_START: 0.040,
        HiddenEvent.COMPLAINT_RESOLVED: 0.600,
    },
    Persona.DORMANT_WEALTHY: {
        HiddenEvent.CARD_DECLINE_SPIKE: 0.005,
        HiddenEvent.CAMPAIGN_EXPOSURE: 0.100,
        HiddenEvent.LOAN_DELINQUENCY_START: 0.002,
        HiddenEvent.COMPLAINT_RESOLVED: 0.700,
    },
    Persona.COMPLAINT_PRONE_CHURNER: {
        HiddenEvent.CARD_DECLINE_SPIKE: 0.020,
        HiddenEvent.CAMPAIGN_EXPOSURE: 0.060,
        HiddenEvent.LOAN_DELINQUENCY_START: 0.010,
        HiddenEvent.COMPLAINT_RESOLVED: 0.500,
    },
}
