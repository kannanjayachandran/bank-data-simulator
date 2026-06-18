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
