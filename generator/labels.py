"""Derived labels and features generator for the synthetic retail banking universe.

Calculates churn labels for prediction horizons (1, 3, 6, 12 months) and compiles
historical feature snapshots for training, ensuring no future data leakage.
"""

from datetime import date
from typing import List, Dict, Any, Optional
import numpy as np
import polars as pl

from config.simulation import SimulationConfig


def generate_churn_labels(
    churn_simulation_state: pl.DataFrame,
    config: SimulationConfig,
) -> pl.DataFrame:
    """Derives customer_churn_label rows post-simulation.

    Grain: (customer_id, as_of_month, prediction_horizon_months)

    Args:
        churn_simulation_state: The final churn simulation state table.
        config: Simulation configuration.

    Returns:
        pl.DataFrame: Churn labels.
    """
    horizons = [1, 3, 6, 12]
    
    # Calculate all simulation months
    months: List[date] = []
    start = config.sim_start
    for i in range(config.sim_months):
        if start.month + i > 12:
            y_offset = (start.month + i - 1) // 12
            m_offset = (start.month + i - 1) % 12 + 1
            curr_month = date(start.year + y_offset, m_offset, 1)
        else:
            curr_month = date(start.year, start.month + i, 1)
        months.append(curr_month)

    label_rows = []

    for row in churn_simulation_state.to_dicts():
        cid = row["customer_id"]
        churn_month = row["churn_month"]  # date or None
        churn_reason = row["churn_reason"]

        for as_of in months:
            # A customer is eligible for labeling if they were active or churned in/after this month
            # Customers who churned strictly before or in this month as_of are NOT labeled for future churn
            if churn_month is not None and churn_month <= as_of:
                continue

            for horizon in horizons:
                # Target range is (as_of, as_of + horizon]
                # Calculate boundary month
                y_offset = (as_of.month + horizon - 1) // 12
                m_offset = (as_of.month + horizon - 1) % 12 + 1
                boundary = date(as_of.year + y_offset, m_offset, 1)

                churned = False
                c_date = None
                c_reason = None

                if churn_month is not None and as_of < churn_month <= boundary:
                    churned = True
                    c_date = churn_month
                    c_reason = churn_reason

                label_rows.append(
                    {
                        "customer_id": cid,
                        "as_of_month": as_of,
                        "prediction_horizon_months": horizon,
                        "churned": churned,
                        "churn_date": c_date,
                        "churn_reason": c_reason,
                    }
                )

    return pl.DataFrame(
        label_rows,
        schema={
            "customer_id": pl.Int64,
            "as_of_month": pl.Date,
            "prediction_horizon_months": pl.Int32,
            "churned": pl.Boolean,
            "churn_date": pl.Date,
            "churn_reason": pl.String,
        },
    )


def add_months(d: date, m: int) -> date:
    """Helper to add months to a date."""
    y_offset = (d.month + m - 1) // 12
    m_offset = (d.month + m - 1) % 12 + 1
    return date(d.year + y_offset, m_offset, 1)


def generate_feature_snapshots(
    customer_master: pl.DataFrame,
    account_master: pl.DataFrame,
    account_monthly_snapshot: pl.DataFrame,
    card_portfolio: pl.DataFrame,
    card_monthly_snapshot: pl.DataFrame,
    loan_master: pl.DataFrame,
    loan_monthly_snapshot: pl.DataFrame,
    product_holdings_monthly: pl.DataFrame,
    customer_monthly_activity: pl.DataFrame,
    digital_engagement_monthly: pl.DataFrame,
    customer_complaints: pl.DataFrame,
    customer_feedback: pl.DataFrame,
    customer_churn_label: pl.DataFrame,
    config: SimulationConfig,
) -> pl.DataFrame:
    """Compiles the churn_feature_snapshot table using native Polars expressions."""
    
    start = config.sim_start
    months_set = {add_months(start, i) for i in range(6, config.sim_months)}
    
    eligible_labels = customer_churn_label.filter(pl.col("as_of_month").is_in(months_set))
    if eligible_labels.is_empty():
        return pl.DataFrame([], schema={
            "customer_id": pl.Int64,
            "as_of_month": pl.Date,
            "prediction_horizon_months": pl.Int32,
            "tenure_months": pl.Int32,
            "products_count": pl.Int32,
            "balance_change_3m": pl.Float64,
            "txn_count_change_3m": pl.Float64,
            "login_count_change_6m": pl.Float64,
            "complaint_count_6m": pl.Int32,
            "unresolved_complaints": pl.Int32,
            "days_since_last_login": pl.Int32,
            "salary_credit_consistency": pl.Float64,
            "credit_utilization": pl.Float64,
            "emi_to_income_ratio": pl.Float64,
            "dormant_days": pl.Int32,
            "nps_avg_12m": pl.Float64,
            "campaign_response_rate": pl.Float64,
            "product_acquisition_velocity_6m": pl.Int32,
            "churned": pl.Boolean,
            "churn_date": pl.Date,
            "churn_reason": pl.String,
        })

    # Unique customer-month pairs for aggregating sparse tables
    cust_months = eligible_labels.select(["customer_id", "as_of_month"]).unique()

    # Join customer details first
    base_df = eligible_labels.join(
        customer_master.select(["customer_id", "customer_since", "annual_income"]),
        on="customer_id",
        how="left"
    )

    # 1. tenure_months
    base_df = base_df.with_columns(
        ((pl.col("as_of_month").dt.year() - pl.col("customer_since").dt.year()) * 12 +
         (pl.col("as_of_month").dt.month() - pl.col("customer_since").dt.month())).cast(pl.Int32).alias("tenure_months")
    )

    # Pre-add customer_id to monthly snapshots if not present
    if "customer_id" not in account_monthly_snapshot.columns:
        account_snapshots = account_monthly_snapshot.join(account_master.select(["account_id", "customer_id"]), on="account_id", how="left")
    else:
        account_snapshots = account_monthly_snapshot

    if "customer_id" not in card_monthly_snapshot.columns:
        card_snapshots = card_monthly_snapshot.join(card_portfolio.select(["card_id", "customer_id"]), on="card_id", how="left")
    else:
        card_snapshots = card_monthly_snapshot

    if "customer_id" not in loan_monthly_snapshot.columns:
        loan_snapshots = loan_monthly_snapshot.join(loan_master.select(["loan_id", "customer_id"]), on="loan_id", how="left")
    else:
        loan_snapshots = loan_monthly_snapshot

    # 2. products_count (at M-1) and product_acquisition_velocity_6m
    prod_m1 = product_holdings_monthly.select([
        "customer_id",
        pl.col("snapshot_month").dt.offset_by("1mo").alias("as_of_month"),
        pl.col("products_count").alias("prod_cnt_m1")
    ])
    prod_m6 = product_holdings_monthly.select([
        "customer_id",
        pl.col("snapshot_month").dt.offset_by("6mo").alias("as_of_month"),
        pl.col("products_count").alias("prod_cnt_m6")
    ])
    base_df = base_df.join(prod_m1, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.join(prod_m6, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.with_columns([
        pl.coalesce(pl.col("prod_cnt_m1"), 1).cast(pl.Int32).alias("products_count"),
        pl.max_horizontal([
            0,
            pl.coalesce(pl.col("prod_cnt_m1"), 0) - pl.coalesce(pl.col("prod_cnt_m6"), 0)
        ]).cast(pl.Int32).alias("product_acquisition_velocity_6m")
    ])

    # 3. balance_change_3m
    cust_monthly_bal = account_snapshots.group_by(["customer_id", "snapshot_month"]).agg(
        pl.col("current_balance").mean().alias("avg_bal")
    )
    for i in range(1, 7):
        bal_mi = cust_monthly_bal.select([
            "customer_id",
            pl.col("snapshot_month").dt.offset_by(f"{i}mo").alias("as_of_month"),
            pl.col("avg_bal").alias(f"bal_m{i}")
        ])
        base_df = base_df.join(bal_mi, on=["customer_id", "as_of_month"], how="left")
    
    base_df = base_df.with_columns([
        pl.coalesce(pl.mean_horizontal(["bal_m1", "bal_m2", "bal_m3"]), 0.0).alias("avg_bal_recent"),
        pl.coalesce(pl.mean_horizontal(["bal_m4", "bal_m5", "bal_m6"]), 0.0).alias("avg_bal_prior"),
    ])
    base_df = base_df.with_columns(
        pl.when(pl.col("avg_bal_prior") > 0)
        .then((pl.col("avg_bal_recent") - pl.col("avg_bal_prior")) / pl.col("avg_bal_prior"))
        .otherwise(0.0)
        .round(4)
        .alias("balance_change_3m")
    )

    # 4. txn_count_change_3m & login_count_change_6m
    cust_monthly_act = customer_monthly_activity.select([
        "customer_id",
        "snapshot_month",
        (pl.col("debit_txn_count") + pl.col("credit_txn_count")).alias("txns"),
        pl.col("login_count").alias("logins"),
        pl.col("days_since_last_txn").alias("dorm_days")
    ])
    for i in range(1, 13):
        act_mi = cust_monthly_act.select([
            "customer_id",
            pl.col("snapshot_month").dt.offset_by(f"{i}mo").alias("as_of_month"),
            pl.col("txns").alias(f"txns_m{i}"),
            pl.col("logins").alias(f"logins_m{i}"),
            *( [pl.col("dorm_days").alias("dorm_days_m1")] if i == 1 else [] )
        ])
        base_df = base_df.join(act_mi, on=["customer_id", "as_of_month"], how="left")

    base_df = base_df.with_columns([
        pl.coalesce(pl.mean_horizontal(["txns_m1", "txns_m2", "txns_m3"]), 0.0).alias("avg_txn_recent"),
        pl.coalesce(pl.mean_horizontal(["txns_m4", "txns_m5", "txns_m6"]), 0.0).alias("avg_txn_prior"),
        pl.coalesce(pl.mean_horizontal([f"logins_m{i}" for i in range(1, 7)]), 0.0).alias("avg_login_recent"),
        pl.coalesce(pl.mean_horizontal([f"logins_m{i}" for i in range(7, 13)]), 0.0).alias("avg_login_prior"),
    ])
    base_df = base_df.with_columns([
        pl.when(pl.col("avg_txn_prior") > 0)
        .then((pl.col("avg_txn_recent") - pl.col("avg_txn_prior")) / pl.col("avg_txn_prior"))
        .otherwise(0.0)
        .round(4)
        .alias("txn_count_change_3m"),
        pl.when(pl.col("avg_login_prior") > 0)
        .then((pl.col("avg_login_recent") - pl.col("avg_login_prior")) / pl.col("avg_login_prior"))
        .otherwise(0.0)
        .round(4)
        .alias("login_count_change_6m"),
        pl.coalesce(pl.col("dorm_days_m1"), 180).cast(pl.Int32).alias("dormant_days")
    ])

    # 5. complaints features
    comp_6m_df = cust_months.join(
        customer_complaints,
        on="customer_id",
        how="inner"
    ).filter(
        (pl.col("complaint_month") >= pl.col("as_of_month").dt.offset_by("-6mo")) &
        (pl.col("complaint_month") < pl.col("as_of_month"))
    ).group_by(["customer_id", "as_of_month"]).len().rename({"len": "comp_count_6m"})

    unres_df = cust_months.join(
        customer_complaints.filter(pl.col("resolved_flag") == False),
        on="customer_id",
        how="inner"
    ).filter(
        pl.col("complaint_month") < pl.col("as_of_month")
    ).group_by(["customer_id", "as_of_month"]).len().rename({"len": "unres_count"})

    base_df = base_df.join(comp_6m_df, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.join(unres_df, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.with_columns([
        pl.coalesce(pl.col("comp_count_6m"), 0).cast(pl.Int32).alias("complaint_count_6m"),
        pl.coalesce(pl.col("unres_count"), 0).cast(pl.Int32).alias("unresolved_complaints")
    ])

    # 6. days_since_last_login
    dig_m1 = digital_engagement_monthly.select([
        "customer_id",
        pl.col("snapshot_month").dt.offset_by("1mo").alias("as_of_month"),
        "last_login_date"
    ])
    base_df = base_df.join(dig_m1, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.with_columns(
        pl.coalesce(
            (pl.col("as_of_month") - pl.col("last_login_date")).dt.total_days(),
            180
        ).cast(pl.Int32).alias("days_since_last_login")
    )

    # 7. salary_credit_consistency
    sal_months = account_snapshots.filter(pl.col("salary_credit_amount") > 0).select(["customer_id", "snapshot_month"]).unique()
    sal_consistency_df = cust_months.join(
        sal_months,
        on="customer_id",
        how="inner"
    ).filter(
        (pl.col("snapshot_month") >= pl.col("as_of_month").dt.offset_by("-6mo")) &
        (pl.col("snapshot_month") < pl.col("as_of_month"))
    ).group_by(["customer_id", "as_of_month"]).len()
    
    base_df = base_df.join(
        sal_consistency_df.select([
            "customer_id",
            "as_of_month",
            (pl.col("len") / 6.0).round(4).alias("salary_credit_consistency")
        ]),
        on=["customer_id", "as_of_month"],
        how="left"
    )
    base_df = base_df.with_columns(
        pl.col("salary_credit_consistency").fill_null(0.0)
    )

    # 8. credit_utilization
    card_util_monthly = card_snapshots.group_by(["customer_id", "snapshot_month"]).agg(
        pl.col("utilization_rate").mean().alias("avg_util")
    )
    card_util_df = card_util_monthly.select([
        "customer_id",
        pl.col("snapshot_month").dt.offset_by("1mo").alias("as_of_month"),
        pl.col("avg_util").alias("credit_utilization")
    ])
    base_df = base_df.join(card_util_df, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.with_columns(
        pl.col("credit_utilization").fill_null(0.0).round(4)
    )

    # 9. emi_to_income_ratio
    loan_emi_monthly = loan_snapshots.group_by(["customer_id", "snapshot_month"]).agg(
        pl.col("emi_amount").sum().alias("total_emi")
    )
    loan_emi_df = loan_emi_monthly.select([
        "customer_id",
        pl.col("snapshot_month").dt.offset_by("1mo").alias("as_of_month"),
        pl.col("total_emi").alias("total_emi")
    ])
    base_df = base_df.join(loan_emi_df, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.with_columns(
        pl.when(pl.col("total_emi").is_not_null())
        .then(pl.col("total_emi") / (pl.col("annual_income") / 12.0))
        .otherwise(0.0)
        .round(4)
        .alias("emi_to_income_ratio")
    )

    # 10. nps_avg_12m
    nps_df = cust_months.join(
        customer_feedback.filter(pl.col("nps_score").is_not_null()),
        on="customer_id",
        how="inner"
    ).filter(
        (pl.col("feedback_month") >= pl.col("as_of_month").dt.offset_by("-12mo")) &
        (pl.col("feedback_month") < pl.col("as_of_month"))
    ).group_by(["customer_id", "as_of_month"]).agg(
        pl.col("nps_score").mean().alias("avg_nps")
    )
    base_df = base_df.join(nps_df, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.with_columns(
        pl.col("avg_nps").fill_null(8.0).round(4).alias("nps_avg_12m")
    )

    # 11. campaign_response_rate
    campaign_df = cust_months.join(
        digital_engagement_monthly,
        on="customer_id",
        how="inner"
    ).filter(
        (pl.col("snapshot_month") >= pl.col("as_of_month").dt.offset_by("-12mo")) &
        (pl.col("snapshot_month") < pl.col("as_of_month"))
    ).group_by(["customer_id", "as_of_month"]).agg([
        pl.col("campaigns_responded").sum().alias("resp"),
        pl.col("campaigns_received").sum().alias("recv")
    ]).select([
        "customer_id",
        "as_of_month",
        pl.when(pl.col("recv") > 0).then(pl.col("resp") / pl.col("recv")).otherwise(0.0).alias("campaign_response_rate")
    ])
    base_df = base_df.join(campaign_df, on=["customer_id", "as_of_month"], how="left")
    base_df = base_df.with_columns(
        pl.col("campaign_response_rate").fill_null(0.0).round(4)
    )

    result = base_df.select([
        "customer_id",
        "as_of_month",
        "prediction_horizon_months",
        "tenure_months",
        "products_count",
        "balance_change_3m",
        "txn_count_change_3m",
        "login_count_change_6m",
        "complaint_count_6m",
        "unresolved_complaints",
        "days_since_last_login",
        "salary_credit_consistency",
        "credit_utilization",
        "emi_to_income_ratio",
        "dormant_days",
        "nps_avg_12m",
        "campaign_response_rate",
        "product_acquisition_velocity_6m",
        "churned",
        "churn_date",
        "churn_reason"
    ])

    return result.cast({
        "customer_id": pl.Int64,
        "as_of_month": pl.Date,
        "prediction_horizon_months": pl.Int32,
        "tenure_months": pl.Int32,
        "products_count": pl.Int32,
        "balance_change_3m": pl.Float64,
        "txn_count_change_3m": pl.Float64,
        "login_count_change_6m": pl.Float64,
        "complaint_count_6m": pl.Int32,
        "unresolved_complaints": pl.Int32,
        "days_since_last_login": pl.Int32,
        "salary_credit_consistency": pl.Float64,
        "credit_utilization": pl.Float64,
        "emi_to_income_ratio": pl.Float64,
        "dormant_days": pl.Int32,
        "nps_avg_12m": pl.Float64,
        "campaign_response_rate": pl.Float64,
        "product_acquisition_velocity_6m": pl.Int32,
        "churned": pl.Boolean,
        "churn_date": pl.Date,
        "churn_reason": pl.String,
    })
