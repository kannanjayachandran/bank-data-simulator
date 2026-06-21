import sys
import os
import line_profiler
from datetime import date
from config.simulation import SimulationConfig
from pipeline.simulate import run_simulation
from generator.labels import generate_feature_snapshots


def main():
    config = SimulationConfig(
        n_customers=2000,
        sim_start=date(2024, 1, 1),
        sim_months=24,
        seed=42
    )

    lp = line_profiler.LineProfiler()
    lp.add_function(run_simulation)
    lp.add_function(generate_feature_snapshots)

    print("Running line-level profiling for 2,000 customers...")
    lp.runcall(run_simulation, config, streaming=False)
    lp.print_stats()


if __name__ == "__main__":
    main()
