from metta.sim.simulation_stats_db import StatsDB


def dump_stats(db: StatsDB, analyzer_cfg: PolicyStatsConfig):
    metrics = analyzer_cfg.metrics

    # Get policy data as pandas DataFrame for easy manipulation
    df = db.query(f"SELECT * FROM policy_simulations_{metrics[0]}")

    if df.empty:
        logger.info(f"No data found for metric '{metric}'")
        return

    # Create a pivot table: policies as rows, environments as columns
    policy_ids = [f"{row['policy_key']}:{row['policy_version']}" for _, row in df.iterrows()]
    df["policy_id"] = policy_ids

    # Basic stats
    logger.info(f"\nDatabase contains {len(df)} policy-environment combinations")
    logger.info(f"Unique policies: {df['policy_id'].nunique()}")
    logger.info(f"Environments: {df['sim_env'].nunique()}")

    # Create pivot table
    pivot = df.pivot_table(index="policy_id", columns="sim_env", values=metric, aggfunc="mean")

    # Add overall average column
    pivot["Overall"] = pivot.mean(axis=1)

    # Sort by overall score
    pivot = pivot.sort_values("Overall")

    # Print the table
    logger.info("\nPolicy Performance Summary:")
    logger.info(pivot.round(4))
