"""Digital engagement monthly generator for the synthetic retail banking universe.

Generates the digital_engagement_monthly DataFrame for a given snapshot month
based on persona digital engagement beta parameters and active campaign events.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional
import numpy as np
import polars as pl

from config.personas import PERSONA_CONFIGS, Persona


def generate_monthly_digital(
    customer_ids: List[int],
    snapshot_month: date,
    personas: List[str],
    rng: np.random.Generator,
    login_counts: Optional[Dict[int, int]] = None,
    days_since_login: Optional[Dict[int, int]] = None,
    active_events_dict: Optional[Dict[int, List[str]]] = None,
) -> pl.DataFrame:
    """Generates the digital_engagement_monthly DataFrame for a given month.

    Args:
        customer_ids: List of customer IDs.
        snapshot_month: Snapshot month (first day of month).
        personas: List of persona strings corresponding to customer_ids.
        rng: Seeded numpy random generator.
        login_counts: Optional dictionary mapping customer_id to their login count of the month.
        days_since_login: Optional dictionary mapping customer_id to days since last login.
        active_events_dict: Optional dictionary mapping customer_id to list of active event names in the month.

    Returns:
        pl.DataFrame: The digital engagement monthly table.
    """
    if login_counts is None:
        login_counts = {}
    if days_since_login is None:
        days_since_login = {}
    if active_events_dict is None:
        active_events_dict = {}

    # Calculate last day of this month for date calculation
    # Handled manually to avoid dependencies
    m = snapshot_month.month
    y = snapshot_month.year
    if m == 12:
        next_m_start = date(y + 1, 1, 1)
    else:
        next_m_start = date(y, m + 1, 1)
    last_day_of_month = next_m_start - timedelta(days=1)

    rows = []
    for cid, pers_name in zip(customer_ids, personas):
        p_enum = Persona(pers_name)
        p_config = PERSONA_CONFIGS[p_enum]
        events = active_events_dict.get(cid, [])
        logins = login_counts.get(cid, 0)
        dsl = days_since_login.get(cid, 30)

        # 1. Sample customer's specific engagement score from their persona's Beta distribution
        engagement_score = rng.beta(
            p_config.digital_engagement_beta_a,
            p_config.digital_engagement_beta_b,
        )

        # 2. Opt-in rate based on persona
        if p_enum == Persona.DIGITAL_NATIVE:
            opt_in_prob = 0.92
        elif p_enum == Persona.DORMANT_WEALTHY:
            opt_in_prob = 0.35
        else:
            opt_in_prob = 0.75
        notification_opt_in = rng.random() < opt_in_prob

        # Active if they logged in during the month
        mobile_app_active = logins > 0 and (
            p_enum != Persona.DORMANT_WEALTHY or rng.random() < 0.6
        )

        # Last login date
        if logins > 0:
            last_login_date = last_day_of_month - timedelta(days=int(dsl))
        else:
            last_login_date = None

        # 3. Notification statistics scaled by engagement score
        if notification_opt_in:
            push_sent = int(rng.poisson(12.0))
            # Open rate scaled by customer engagement score
            push_opened = int(
                rng.binomial(push_sent, max(0.05, min(0.95, engagement_score)))
            )
        else:
            push_sent = 0
            push_opened = 0

        email_sent = int(rng.poisson(6.0))
        # Email click rate is lower than push notifications open rate
        email_clicks = int(
            rng.binomial(email_sent, max(0.01, min(0.40, engagement_score * 0.3)))
        )

        # 4. Campaigns response mapping
        campaigns_received = 0
        campaigns_responded = 0
        if "campaign_exposure" in events:
            campaigns_received = 1
            # Response rate driven by engagement score and campaign success modifiers
            response_prob = max(0.05, min(0.90, engagement_score))
            if rng.random() < response_prob:
                campaigns_responded = 1

        # Web sessions
        web_sessions = int(rng.poisson(max(0.5, engagement_score * 8.0)))

        # App crashes (Poissons)
        crash_count = int(rng.poisson(0.15))

        rows.append(
            {
                "customer_id": cid,
                "snapshot_month": snapshot_month,
                "mobile_app_active": mobile_app_active,
                "last_login_date": last_login_date,
                "push_notifications_sent": push_sent,
                "push_notifications_opened": push_opened,
                "email_campaigns_sent": email_sent,
                "email_clicks": email_clicks,
                "campaigns_received": campaigns_received,
                "campaigns_responded": campaigns_responded,
                "web_sessions": web_sessions,
                "notification_opt_in": notification_opt_in,
                "app_crash_count": crash_count,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "customer_id": pl.Int64,
            "snapshot_month": pl.Date,
            "mobile_app_active": pl.Boolean,
            "last_login_date": pl.Date,
            "push_notifications_sent": pl.Int32,
            "push_notifications_opened": pl.Int32,
            "email_campaigns_sent": pl.Int32,
            "email_clicks": pl.Int32,
            "campaigns_received": pl.Int32,
            "campaigns_responded": pl.Int32,
            "web_sessions": pl.Int32,
            "notification_opt_in": pl.Boolean,
            "app_crash_count": pl.Int32,
        },
    )
