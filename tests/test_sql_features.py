"""Test that the features materialized by build_features.sql match the Python baseline exactly."""

import os
import duckdb
import polars as pl
from datetime import date
from config.simulation import SimulationConfig
from pipeline.simulate import run_simulation


def test_sql_features_match_python():
    # 1. Run a small simulation
    config = SimulationConfig(n_customers=100, sim_months=12, seed=42)
    results = run_simulation(config, streaming=False)
    
    # 2. Setup in-memory DuckDB
    con = duckdb.connect()
    
    # Create and load all tables
    for name, df in results.items():
        con.register(f"{name}_temp", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_temp")
        con.unregister(f"{name}_temp")
        
    # Rename python-generated features to baseline
    con.execute("ALTER TABLE churn_feature_snapshot RENAME TO churn_feature_snapshot_python_baseline")
    
    # Create empty churn_feature_snapshot table (using same schema)
    con.execute("CREATE TABLE churn_feature_snapshot AS SELECT * FROM churn_feature_snapshot_python_baseline WHERE FALSE")
    
    # 3. Read features/build_features.sql DDL and execute it
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql_path = os.path.join(base_dir, "features", "build_features.sql")
    assert os.path.exists(sql_path), f"SQL file not found at {sql_path}"
    
    with open(sql_path, "r") as f:
        sql_content = f.read()
        
    # Execute statements split by semicolon
    for stmt in sql_content.split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)
            
    # 4. Compare SQL-generated features with python-generated features
    sql_features = con.execute("""
        SELECT * 
        FROM churn_feature_snapshot 
        ORDER BY customer_id, as_of_month, prediction_horizon_months
    """).pl()
    
    py_features = con.execute("""
        SELECT * 
        FROM churn_feature_snapshot_python_baseline 
        ORDER BY customer_id, as_of_month, prediction_horizon_months
    """).pl()
    
    # Check that lengths match
    assert len(sql_features) == len(py_features), f"Row count mismatch: SQL={len(sql_features)}, Python={len(py_features)}"
    
    # Compare each column individually to help debug in case of differences
    for col in py_features.columns:
        sql_col = sql_features[col]
        py_col = py_features[col]
        
        # Check type
        assert sql_col.dtype == py_col.dtype, f"Column '{col}' type mismatch: SQL={sql_col.dtype}, Python={py_col.dtype}"
        
        # Check values
        if py_col.dtype in (pl.Float64, pl.Float32):
            # For floats, handle minor precision/rounding differences or null comparisons
            for idx, (s_val, p_val) in enumerate(zip(sql_col, py_col)):
                if s_val is None or p_val is None:
                    assert s_val == p_val, f"Null mismatch in '{col}' at index {idx}: SQL={s_val}, Python={p_val}"
                else:
                    assert abs(s_val - p_val) < 2e-4, f"Value mismatch in '{col}' at index {idx}: SQL={s_val}, Python={p_val}"
        else:
            assert sql_col.equals(py_col), f"Values in column '{col}' mismatch!"
            
    # All columns matched successfully!
    print("✔ SQL features match Python baseline exactly (within float precision tolerances).")
