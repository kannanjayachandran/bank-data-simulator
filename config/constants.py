from typing import Tuple

# Randomization defaults
DEFAULT_SEED: int = 42

# ID generation start offsets
CUSTOMER_ID_START: int = 1_000_000
ACCOUNT_ID_START: int = 10_000_000
CARD_ID_START: int = 20_000_000
LOAN_ID_START: int = 30_000_000
TRANSACTION_ID_START: int = 40_000_000
COMPLAINT_ID_START: int = 50_000_000
FEEDBACK_ID_START: int = 60_000_000

# Geographic and localization defaults
DEFAULT_CURRENCY: str = "INR"
DEFAULT_TIMEZONE: str = "UTC"

# Low-sensitivity segment share range
# The generator spine will sample a share from this range for the customer pool.
LOW_SENSITIVITY_SHARE_RANGE: Tuple[float, float] = (0.10, 0.15)

# Card and credit defaults
CREDIT_LIMIT_MULTIPLIER: float = 3.0
PRIMARY_ACCOUNT_TYPE: str = "Savings"

# Loan tenure ranges (in months)
PERSONAL_LOAN_TENURE_RANGE: Tuple[int, int] = (12, 60)
HOME_LOAN_TENURE_RANGE: Tuple[int, int] = (120, 240)

# Branch Selection Weights
BRANCH_METRO_WEIGHT: float = 2.0
BRANCH_URBAN_WEIGHT: float = 1.0

# Fixed Obligation to Income Ratio (FOIR) Limits
FOIR_LIMITS = {
    "Personal Loan": 0.40,
    "Home Loan": 0.50
}

# Minimum loan amounts for FOIR checking
MIN_PERSONAL_LOAN_AMOUNT: float = 50_000.0
MIN_HOME_LOAN_AMOUNT: float = 2_000_000.0



