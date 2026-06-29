"""Customer master generator for the synthetic retail banking universe.

Generates the core demography, income, and locations of customers using Faker
and statistical distributions.
"""

from datetime import date
from typing import Optional

import numpy as np
import polars as pl
from faker import Faker

from config.personas import PERSONA_CONFIGS, Persona
from config.simulation import SimulationConfig
from generator.branches import generate_branches
from generator.spine import Spine


def generate_customers(
    spine: Spine,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
    branches_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Generates the static customer_master DataFrame from the spine.

    Args:
        spine: Spine containing customer IDs and personas.
        config: Simulation configuration.
        branches_df: Optional pre-generated branch master DataFrame.
                     If None, branches are generated internally.
                     Pass the existing DataFrame to avoid regenerating.
        rng: Optional seeded numpy random generator.

    Returns:
        pl.DataFrame: The customer master table.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    if branches_df is None:
        branches_df = generate_branches()

    # Initialize Faker with Indian English locale
    fake = Faker("en_IN")
    Faker.seed(config.seed)

    state_df = spine.simulation_state
    customer_ids = state_df["customer_id"].to_list()
    personas = state_df["persona"].to_list()

    # --- Branch weight resolution ---
    # Read customer_weight directly from branch data instead of hardcoded
    # constants. This keeps weight logic co-located with branch definitions
    # and automatically picks up any changes made in branches.py.
    branch_codes = branches_df["branch_code"].to_list()
    branch_cities = branches_df["city"].to_list()
    branch_states = branches_df["state"].to_list()
    raw_weights = branches_df["customer_weight"].to_numpy().astype(float)
    normalized_weights = raw_weights / raw_weights.sum()

    # Vectorized branch assignment across all customers in one RNG call
    branch_indices = rng.choice(
        len(branch_codes), size=config.n_customers, p=normalized_weights
    )

    # Pre-assign branch codes, cities, states from the branch index
    # Avoids repeated list lookups inside the customer loop
    assigned_branch_codes = [branch_codes[i] for i in branch_indices]
    assigned_cities = [branch_cities[i] for i in branch_indices]
    assigned_states = [branch_states[i] for i in branch_indices]

    # --- Vectorized draws for fields that don't depend on per-customer state ---
    # Drawing in batch is significantly faster than per-customer rng calls.
    # Fields that depend on earlier draws (e.g. marital status depends on age)
    # remain in the loop.
    n = config.n_customers
    gender_draws = rng.choice([0, 1], size=n, p=[0.52, 0.48])  # 0=Male, 1=Female
    age_draws = np.clip(rng.normal(38.0, 12.0, size=n), 18, 75).astype(int)
    birth_months = rng.integers(1, 13, size=n)
    birth_days = rng.integers(1, 29, size=n)
    kyc_draws = rng.choice(
        ["Verified", "Pending", "Failed"], size=n, p=[0.95, 0.04, 0.01]
    )
    income_log_mus = np.array(
        [PERSONA_CONFIGS[Persona(p)].income_log_mu for p in personas]
    )
    income_log_sigmas = np.array(
        [PERSONA_CONFIGS[Persona(p)].income_log_sigma for p in personas]
    )
    income_clip_mins = np.array(
        [PERSONA_CONFIGS[Persona(p)].income_clip_min for p in personas]
    )
    income_clip_maxs = np.array(
        [PERSONA_CONFIGS[Persona(p)].income_clip_max for p in personas]
    )
    # Single vectorized lognormal draw for all customers
    raw_incomes = rng.lognormal(income_log_mus, income_log_sigmas)
    incomes = np.clip(raw_incomes, income_clip_mins, income_clip_maxs).round(2)

    # Tenure: exponential right-skewed, clipped to [1, 120] months
    tenure_months_arr = np.clip(rng.exponential(36.0, size=n), 1, 120).astype(int)

    # Marital status draws — two pools, selected by age in the loop
    marital_young = rng.choice(["Single", "Married"], size=n, p=[0.80, 0.20])
    marital_adult = rng.choice(
        ["Married", "Single", "Divorced", "Widowed"],
        size=n,
        p=[0.85, 0.10, 0.04, 0.01],
    )

    # Occupation pools per persona — drawn in batch, selected by persona in loop
    salaried_occupations = rng.choice(
        [
            "Software Engineer",
            "Sales Executive",
            "Teacher",
            "Manager",
            "Bank Officer",
            "Analyst",
        ],
        size=n,
    )
    affluent_occupations = rng.choice(
        ["Doctor", "Business Owner", "Consultant", "Director", "Lawyer"],
        size=n,
    )
    dormant_occupations = rng.choice(
        ["Retired", "Investor", "Business Owner", "Consultant"],
        size=n,
    )
    affluent_emp_types = rng.choice(
        ["Self-Employed", "Professional"], size=n, p=[0.60, 0.40]
    )
    dormant_emp_types = rng.choice(
        ["Retired", "Self-Employed", "Professional"],
        size=n,
        p=[0.70, 0.20, 0.10],
    )

    # Salaried personas — names drawn in batch separated by gender
    male_mask = gender_draws == 0
    n_male = int(male_mask.sum())
    n_female = n - n_male

    male_first_names = [fake.first_name_male() for _ in range(n_male)]
    female_first_names = [fake.first_name_female() for _ in range(n_female)]
    last_names = [fake.last_name() for _ in range(n)]

    # Assign names by gender index
    male_iter = iter(male_first_names)
    female_iter = iter(female_first_names)
    first_names = [
        next(male_iter) if male_mask[i] else next(female_iter) for i in range(n)
    ]

    # Salaried persona set — used for occupation/employment branching
    salaried_personas = {
        Persona.SALARY_CORE,
        Persona.DIGITAL_NATIVE,
        Persona.CREDIT_STRESSED,
        Persona.COMPLAINT_PRONE_CHURNER,
    }

    rows = []
    for idx, (cid, pers_name) in enumerate(zip(customer_ids, personas)):
        p_enum = Persona(pers_name)

        # Name and gender
        gender = "Male" if male_mask[idx] else "Female"
        first_name = first_names[idx]
        last_name = last_names[idx]

        # Age and date of birth
        age = age_draws[idx]
        birth_year = config.sim_start.year - age
        date_of_birth = date(birth_year, int(birth_months[idx]), int(birth_days[idx]))

        # Marital status — depends on age drawn above
        marital_status = marital_young[idx] if age < 25 else marital_adult[idx]

        # Occupation and employment type — depends on persona
        if p_enum in salaried_personas:
            occupation = salaried_occupations[idx]
            employment_type = "Salaried"
        elif p_enum == Persona.AFFLUENT_MULTI_PRODUCT:
            occupation = affluent_occupations[idx]
            employment_type = affluent_emp_types[idx]
        else:  # DORMANT_WEALTHY
            occupation = dormant_occupations[idx]
            employment_type = dormant_emp_types[idx]

        # Income — already vectorized above
        income = float(incomes[idx])

        # Customer since — derived from tenure
        tenure_months = tenure_months_arr[idx]
        since_year = config.sim_start.year - (tenure_months // 12)
        since_month = config.sim_start.month - (tenure_months % 12)
        if since_month <= 0:
            since_year -= 1
            since_month += 12
        customer_since = date(since_year, since_month, 1)

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
                "city": assigned_cities[idx],
                "state": assigned_states[idx],
                "branch_code": assigned_branch_codes[idx],
                "country": "India",
                "kyc_status": kyc_draws[idx],
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
            "branch_code": pl.String,
            "country": pl.String,
            "kyc_status": pl.String,
            "is_active": pl.Boolean,
        },
    )
