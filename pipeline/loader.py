"""PostgreSQL bulk loader library for the synthetic retail banking universe.

Uses psycopg2 COPY FROM STDIN for fast loading and handles database unreachable
states gracefully.
"""

import io
import os
from typing import Dict
import polars as pl

try:
    import psycopg2
except ImportError:
    psycopg2 = None


def load_to_postgres(dataframes: Dict[str, pl.DataFrame], connection_uri: str) -> None:
    """Loads final Polars DataFrames into a PostgreSQL database.

    Reads the database schema from pipeline/schema.sql, initializes the database,
    and bulk loads the data using psycopg2 copy_expert. Gracefully skips if database
    is unreachable or psycopg2 is not installed.

    Args:
        dataframes: Dictionary mapping table names to Polars DataFrames.
        connection_uri: PostgreSQL connection string.
    """
    if psycopg2 is None:
        print("Warning: psycopg2 is not installed. Skipping database load.")
        return

    # Try connecting to PostgreSQL
    try:
        conn = psycopg2.connect(connection_uri)
    except Exception as e:
        print(
            f"Warning: Could not connect to PostgreSQL database ({e}). Skipping database load."
        )
        return

    try:
        cursor = conn.cursor()

        # Find schema.sql path
        # Assume it's located at pipeline/schema.sql relative to workspace root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_path = os.path.join(base_dir, "pipeline", "schema.sql")

        if not os.path.exists(schema_path):
            schema_path = os.path.join(
                os.path.abspath(os.curdir), "pipeline", "schema.sql"
            )

        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                ddl = f.read()
            cursor.execute(ddl)
            conn.commit()
            print("Database tables initialized from schema.sql.")
        else:
            print("Warning: schema.sql not found. Skipping table initialization.")

        # Order tables to respect Foreign Key dependencies
        load_order = [
            "branch_master",
            "customer_master",
            "account_master",
            "card_portfolio",
            "loan_master",
            "account_monthly_snapshot",
            "card_monthly_snapshot",
            "loan_monthly_snapshot",
            "product_holdings_monthly",
            "transaction_fact",
            "customer_monthly_activity",
            "digital_engagement_monthly",
            "customer_complaints",
            "customer_feedback",
            "churn_simulation_state",
            "customer_churn_label",
            "churn_feature_snapshot",
        ]

        for name in load_order:
            if name not in dataframes:
                continue

            df = dataframes[name]
            if df.is_empty():
                continue

            # Convert Polars DataFrame to CSV in-memory bytes buffer
            buffer = io.BytesIO()
            # Set null_value="" to match NULL '' in COPY statement
            df.write_csv(buffer, include_header=False, separator=",", null_value="")
            buffer.seek(0)

            # COPY command
            copy_sql = (
                f"COPY {name} FROM STDIN WITH (FORMAT csv, HEADER false, NULL '')"
            )
            try:
                cursor.copy_expert(copy_sql, buffer)
                conn.commit()
                print(f"Successfully bulk loaded {df.height} rows into {name}.")
            except Exception as load_err:
                conn.rollback()
                print(
                    f"Error bulk loading into {name}: {load_err}. Skipping this table."
                )

    except Exception as run_err:
        print(f"Error during PostgreSQL processing: {run_err}")
    finally:
        conn.close()
