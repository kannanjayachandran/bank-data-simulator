import argparse
from datetime import date
import os
import shutil
import time
from config.simulation import SimulationConfig
from pipeline.simulate import run_simulation
from pipeline.writer import write_to_parquet
from pipeline.loader import load_to_postgres
import duckdb


def main():
    parser = argparse.ArgumentParser(description="Churn Compass Bank Data Simulator CLI")
    parser.add_argument("--n-customers", type=int, default=1000, help="Number of customers to generate")
    parser.add_argument("--sim-months", type=int, default=24, help="Number of simulation months")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output-dir", default="./data/raw", help="Output directory for Parquet files")
    parser.add_argument("--postgres-uri", help="PostgreSQL connection URI (optional)")
    parser.add_argument("--duckdb-db", help="DuckDB file path to load data (optional)")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs to run")
    
    args = parser.parse_args()
    
    config = SimulationConfig(
        n_customers=args.n_customers,
        sim_start=date(2024, 1, 1),
        sim_months=args.sim_months,
        seed=args.seed
    )
    
    # 1. Run simulation
    print(f"Running simulation for {config.n_customers} customers over {config.sim_months} months with {args.jobs} job(s)...")
    start_time = time.time()
    results = run_simulation(config, streaming=False, jobs=args.jobs)
    sim_duration = time.time() - start_time
    print(f"✔ Simulation completed in {sim_duration:.2f} seconds.")
    
    # Clean output dir
    if os.path.exists(args.output_dir):
        print(f"Cleaning old parquet directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Save Parquet
    print(f"Saving partitioned Parquet files to: {args.output_dir}...")
    write_to_parquet(results, args.output_dir)
    print("✔ Parquet write completed.")
    
    # 3. Load to Postgres if requested
    if args.postgres_uri:
        print(f"Loading to PostgreSQL at: {args.postgres_uri}...")
        load_to_postgres(results, args.postgres_uri)
        print("✔ PostgreSQL load completed.")
        
    # 4. Load to DuckDB if requested
    if args.duckdb_db:
        print(f"Loading to DuckDB at: {args.duckdb_db}...")
        if os.path.exists(args.duckdb_db):
            print(f"Cleaning old database file: {args.duckdb_db}")
            os.remove(args.duckdb_db)
        os.makedirs(os.path.dirname(args.duckdb_db), exist_ok=True)
        con = duckdb.connect(args.duckdb_db)
        
        static_tables = [
            "branch_master", "customer_master", "account_master", 
            "card_portfolio", "loan_master", "churn_simulation_state"
        ]
        partitioned_tables = [
            "account_monthly_snapshot", "card_monthly_snapshot", "loan_monthly_snapshot",
            "product_holdings_monthly", "transaction_fact", "customer_monthly_activity",
            "digital_engagement_monthly", "customer_churn_label", "churn_feature_snapshot",
            "customer_complaints", "customer_feedback"
        ]
        
        for t in static_tables:
            path = os.path.join(args.output_dir, f"{t}.parquet")
            con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{path}')")
            print(f"  ✔ Loaded static table: {t}")
        
        for t in partitioned_tables:
            glob = os.path.join(args.output_dir, t, "**", "*.parquet")
            con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)")
            print(f"  ✔ Loaded partitioned table: {t}")
            
        con.close()
        print("✔ DuckDB load completed.")


if __name__ == "__main__":
    main()
