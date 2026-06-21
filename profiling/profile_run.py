import time
import cProfile
import pstats
import io
from datetime import date
from config.simulation import SimulationConfig
from pipeline.simulate import run_simulation
from pipeline.writer import write_to_parquet


def profile_simulation():
    config = SimulationConfig(
        n_customers=2000,
        sim_start=date(2024, 1, 1),
        sim_months=24,
        seed=42
    )

    print("======================================================================")
    # 1. Component-level timing
    print("[1/2] RUNNING TIME BREAKDOWN...")
    print("======================================================================")
    
    t0 = time.time()
    # Let's break down run_simulation phases if possible, or run E2E
    start_sim = time.time()
    results = run_simulation(config, streaming=False)
    sim_duration = time.time() - start_sim
    print(f"✔ E2E Simulation completed in {sim_duration:.3f} seconds.")
    
    start_write = time.time()
    write_to_parquet(results, "./data/raw_profile")
    write_duration = time.time() - start_write
    print(f"✔ Parquet write completed in {write_duration:.3f} seconds.")
    
    # 2. cProfile for function-level hotspots
    print("\n======================================================================")
    print("[2/2] RUNNING FUNCTION-LEVEL PROFILE (cProfile)...")
    print("======================================================================")
    
    pr = cProfile.Profile()
    pr.enable()
    
    run_simulation(config, streaming=False)
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(35)  # Print top 35 functions
    print(s.getvalue())


if __name__ == "__main__":
    profile_simulation()
