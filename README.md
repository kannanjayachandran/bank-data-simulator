# Synthetic Banking Data Generator

This repository contains the synthetic retail banking data generator used to support Churn Compass. It creates a realistic synthetic banking universe to provide data for downstream models.

## Scope

The generator produces the following datasets:
- Customer master data
- Account and card portfolios
- Loan lifecycle snapshots
- Monthly activity and digital engagement
- Complaints and feedback
- Churn ground truth and derived churn labels for modeling

## Core Design Rules
- **Grain Rules**: Strict one-to-one and one-to-many relationships are maintained across entities like customers, accounts, cards, loans, transactions, and monthly snapshots.
- **Time Handling**: Snapshots are generated monthly (using the first day of the month), and transaction timestamps are tracked strictly in UTC without leaking future data into earlier snapshots.
- **Simulation Principle**: A hidden customer spine is generated first, followed by monthly behavior simulation derived from the spine. Churn events and labels are derived during the simulation process.

## Persona Model
The generator leverages 6 unique personas to model diverse retail banking behaviors:
- `salary_core`
- `affluent_multi_product`
- `digital_native`
- `credit_stressed`
- `dormant_wealthy`
- `complaint_prone_churner`

Each persona defines unique attributes such as annual income ranges, product uptake probabilities, digital engagement, complaint propensity, and base monthly churn rates.

## Event Model
A robust hidden event model governs state transitions:
- **Unconditional Events**: Pre-assigned during spine generation (e.g., job change, large life expense).
- **Conditional Events**: Resolved dynamically month-by-month based on prerequisites (e.g., card decline spike, loan delinquency start).

## Project Status

Current status: **Phase 2 complete**.

### Phase 1: Config
- Core configuration system implemented.
- Persona parameters, event probability tables, and simulation configs set up.
- Validated via automated unit tests.

### Phase 2: Spine + Churn Scoring
- `generators/spine.py` completed: assigns `customer_id`, `persona`, segment types, and pre-schedules unconditional events.
- `generators/churn.py` completed: handles score formula, sigmoids, hard triggers, and churn reason priority logic.

## Technology Stack
- **Languages**: Python 3.13+
- **Core Libraries**: NumPy, SciPy, Faker, PyArrow, Polars, Pydantic
- **Data & Storage**: DuckDB, PostgreSQL
- **Infrastructure**: Docker / Docker Compose

For a complete breakdown of rules, constraints, and schemas, please refer to the `implementation-spec.md`.
