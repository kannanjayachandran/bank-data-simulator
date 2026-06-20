"""Parquet writer library for the synthetic retail banking universe.

Saves master tables as single Parquet files and partitioned monthly tables.
"""

import os
from typing import Dict
import polars as pl


def write_to_parquet(dataframes: Dict[str, pl.DataFrame], output_dir: str) -> None:
    """Saves generated Polars DataFrames to the output directory.

    Static tables are written as single files. Monthly snapshot and transaction
    tables are partitioned by month.

    Args:
        dataframes: Dictionary mapping table names to Polars DataFrames.
        output_dir: Root output directory path.
    """
    os.makedirs(output_dir, exist_ok=True)

    static_tables = {
        "branch_master",
        "customer_master",
        "account_master",
        "card_portfolio",
        "loan_master",
        "churn_simulation_state",
    }

    # Partition column mappings
    partition_cols = {
        "account_monthly_snapshot": "snapshot_month",
        "card_monthly_snapshot": "snapshot_month",
        "loan_monthly_snapshot": "snapshot_month",
        "product_holdings_monthly": "snapshot_month",
        "transaction_fact": "txn_month",
        "customer_monthly_activity": "snapshot_month",
        "digital_engagement_monthly": "snapshot_month",
        "customer_churn_label": "as_of_month",
        "churn_feature_snapshot": "as_of_month",
        "customer_complaints": "complaint_month",
        "customer_feedback": "feedback_month",
    }

    for name, df in dataframes.items():
        if df.is_empty():
            continue

        if name in static_tables:
            # Write static master tables as single files
            file_path = os.path.join(output_dir, f"{name}.parquet")
            df.write_parquet(file_path)
        elif name in partition_cols:
            # Write monthly snapshots partitioned by date
            part_col = partition_cols[name]
            # Ensure partition column is of Date or String type
            unique_vals = df[part_col].unique().sort().to_list()
            for val in unique_vals:
                sub_df = df.filter(pl.col(part_col) == val)
                val_str = val.strftime("%Y-%m-%d") if hasattr(val, "strftime") else str(val)
                part_dir = os.path.join(output_dir, name, f"{part_col}={val_str}")
                os.makedirs(part_dir, exist_ok=True)
                file_path = os.path.join(part_dir, "part-0.parquet")
                sub_df.write_parquet(file_path)
        else:
            # Fallback for any other table
            file_path = os.path.join(output_dir, f"{name}.parquet")
            df.write_parquet(file_path)
