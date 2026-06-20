"""Customer master generator for the synthetic retail banking universe.

Generates the core demography, income, and locations of customers using Faker
and statistical distributions.
"""

from datetime import date
from typing import Optional
from faker import Faker
import numpy as np
import polars as pl

from config.constants import (
    BRANCH_METRO_WEIGHT,
    BRANCH_URBAN_WEIGHT,
)
from config.personas import PERSONA_CONFIGS, Persona
from config.simulation import SimulationConfig
from generator.spine import Spine


# Branch location mapping with selection weights
CITIES_STATES = [
    ("Mumbai", "Maharashtra", "West", "Metro"),
    ("Bengaluru", "Karnataka", "South", "Metro"),
    ("Delhi", "Delhi", "North", "Metro"),
    ("Kolkata", "West Bengal", "East", "Metro"),
    ("Chennai", "Tamil Nadu", "South", "Metro"),
    ("Pune", "Maharashtra", "West", "Urban"),
    ("Hyderabad", "Telangana", "South", "Urban"),
    ("Ahmedabad", "Gujarat", "West", "Urban"),
    ("Lucknow", "Uttar Pradesh", "North", "Urban"),
    ("Jaipur", "Rajasthan", "West", "Urban"),
]


def generate_customers(
    spine: Spine,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Generates the static customer_master DataFrame from the spine.

    Args:
        spine: Spine containing customer IDs and personas.
        config: Simulation configuration.
        rng: Optional seeded numpy random generator.

    Returns:
        pl.DataFrame: The customer master table.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    # Initialize Faker with Indian English locale for realistic names and addresses
    fake = Faker("en_IN")
    # Feed seed to faker to maintain determinism
    Faker.seed(config.seed)

    state_df = spine.simulation_state
    customer_ids = state_df["customer_id"].to_list()
    personas = state_df["persona"].to_list()

    # Pre-calculate branch weights
    weights = []
    for item in CITIES_STATES:
        w = BRANCH_METRO_WEIGHT if item[3] == "Metro" else BRANCH_URBAN_WEIGHT
        weights.append(w)
    weights = np.array(weights) / sum(weights)

    # Pre-select indices of locations using weighted distribution
    location_indices = rng.choice(len(CITIES_STATES), size=config.n_customers, p=weights)

    rows = []
    for idx, (cid, pers_name) in enumerate(zip(customer_ids, personas)):
        p_enum = Persona(pers_name)
        p_config = PERSONA_CONFIGS[p_enum]

        # 1. Generate name and gender
        gender = rng.choice(["Male", "Female"], p=[0.52, 0.48])
        if gender == "Male":
            first_name = fake.first_name_male()
        else:
            first_name = fake.first_name_female()
        last_name = fake.last_name()

        # 2. Age - Truncated Normal distribution [18, 75]
        age = rng.normal(38.0, 12.0)
        age = int(np.clip(age, 18, 75))
        birth_year = config.sim_start.year - age
        birth_month = rng.integers(1, 13)
        birth_day = rng.integers(1, 29)  # Safe limit for all months
        date_of_birth = date(birth_year, birth_month, birth_day)

        # 3. Marital status based on age
        if age < 25:
            marital_status = rng.choice(["Single", "Married"], p=[0.80, 0.20])
        else:
            marital_status = rng.choice(
                ["Married", "Single", "Divorced", "Widowed"],
                p=[0.85, 0.10, 0.04, 0.01],
            )

        # 4. Occupation & employment type based on Persona
        if p_enum in [
            Persona.SALARY_CORE,
            Persona.DIGITAL_NATIVE,
            Persona.CREDIT_STRESSED,
            Persona.COMPLAINT_PRONE_CHURNER,
        ]:
            occupation = rng.choice(
                ["Software Engineer", "Sales Executive", "Teacher", "Manager", "Bank Officer", "Analyst"]
            )
            employment_type = "Salaried"
        elif p_enum == Persona.AFFLUENT_MULTI_PRODUCT:
            occupation = rng.choice(
                ["Doctor", "Business Owner", "Consultant", "Director", "Lawyer"]
            )
            employment_type = rng.choice(["Self-Employed", "Professional"], p=[0.60, 0.40])
        else:  # Persona.DORMANT_WEALTHY
            occupation = rng.choice(["Retired", "Investor", "Business Owner", "Consultant"])
            employment_type = rng.choice(
                ["Retired", "Self-Employed", "Professional"], p=[0.70, 0.20, 0.10]
            )

        # 5. Income - lognormal draw and clip
        income = rng.lognormal(p_config.income_log_mu, p_config.income_log_sigma)
        income = round(float(np.clip(income, p_config.income_clip_min, p_config.income_clip_max)), 2)

        # 6. Customer tenure (months) - exponential right-skewed distribution
        tenure_months = rng.exponential(36.0)
        tenure_months = max(1, min(120, int(tenure_months)))
        
        # Compute customer_since subtracting tenure_months from sim_start
        since_year = config.sim_start.year - (tenure_months // 12)
        since_month = config.sim_start.month - (tenure_months % 12)
        if since_month <= 0:
            since_year -= 1
            since_month += 12
        customer_since = date(since_year, since_month, 1)

        # 7. Geographic location from weighted selection
        loc_idx = location_indices[idx]
        city, state, _, _ = CITIES_STATES[loc_idx]

        # 8. KYC status Verification rate
        kyc_status = rng.choice(["Verified", "Pending", "Failed"], p=[0.95, 0.04, 0.01])

        rows.append(
            {
                "customer_id": cid,
                "cif_number": f"CIF{cid}",
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": date_of_birth,
                "gender": gender,
                "marital_status": marital_status,
                "occupation": occupation,
                "employment_type": employment_type,
                "annual_income": income,
                "customer_since": customer_since,
                "city": city,
                "state": state,
                "country": "India",
                "kyc_status": kyc_status,
                "is_active": True,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "customer_id": pl.Int64,
            "cif_number": pl.String,
            "first_name": pl.String,
            "last_name": pl.String,
            "date_of_birth": pl.Date,
            "gender": pl.String,
            "marital_status": pl.String,
            "occupation": pl.String,
            "employment_type": pl.String,
            "annual_income": pl.Float64,
            "customer_since": pl.Date,
            "city": pl.String,
            "state": pl.String,
            "country": pl.String,
            "kyc_status": pl.String,
            "is_active": pl.Boolean,
        },
    )
