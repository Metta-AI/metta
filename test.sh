#!/bin/bash
echo "Running navigation eval debug test"
RANDOM_NUM=$((RANDOM % 1000))
IDX="eval_test_${RANDOM_NUM}"
DB_PATH="./train_dir/navigation${IDX}/stats.db"

echo "Running simulation..."
python3 -m tools.sim \
run=navigation$IDX \
device=cpu \
policy_uri=wandb://run/b.daphne.navigation0:v12 \
+eval_db_uri=wandb://artifacts/navigation_db \
sim_job.policy_uris=[wandb://run/b.daphne.navigation] \
+sim_job.simulation_suite.name=navigation \
+sim_job.simulation_suite.num_episodes=1 \
+sim_job.simulation_suite.max_time_s=60 \
+sim_job.simulation_suite.simulations={navigation/wanderout:{env:env/mettagrid/navigation/evals/wanderout}}

echo "Simulation completed"
echo "Database created at: $DB_PATH"

# Check if the database was created successfully
if [ -f "$DB_PATH" ]; then
    echo "Analyzing reward statistics..."
    
    # Run DuckDB query and store results
    REWARD_STATS=$(duckdb "$DB_PATH" -c "
    SELECT
        MIN(value) AS min_reward,
        MAX(value) AS max_reward,
        AVG(value) AS avg_reward,
        COUNT(*) AS count
    FROM agent_metrics
    WHERE metric = 'reward';
    ")
    
    echo "======== REWARD STATISTICS ========"
    echo "$REWARD_STATS"
    echo "=================================="
    
    # Optionally, you can run additional queries
    echo "Action statistics by type:"
    duckdb "$DB_PATH" -c "
    SELECT 
        metric,
        MIN(value) AS min_value,
        MAX(value) AS max_value,
        AVG(value) AS avg_value,
        COUNT(*) AS count
    FROM agent_metrics
    WHERE metric LIKE 'action.%'
    GROUP BY metric
    ORDER BY metric;
    "
else
    echo "Error: Database file was not created at $DB_PATH"
fi