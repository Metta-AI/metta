#!/usr/bin/env python
import argparse
from wan_duckdb import WanDuckDb  # Assuming you save the WanDuckDb class in wan_duckdb.py

def main():
    parser = argparse.ArgumentParser(description="Score Policies based on Eval Data")
    parser.add_argument("--policies", type=str, required=True,
                        help="Comma-separated list of policies to evaluate, e.g., a,b,c")
    parser.add_argument("--baseline", type=str, required=True,
                        help="The baseline policy name")
    parser.add_argument("--evals", type=str, required=True,
                        help="Comma-separated list of eval names to include, e.g., simple_small,simple_medium,mazes")
    parser.add_argument("--glicko2", action="store_true",
                        help="Run Glicko2 rating updates")
    parser.add_argument("--metric", type=str, required=True,
                        help="The metric to query, e.g., action.altar.use")
    parser.add_argument("--artifact", type=str, default="eval_db",
                        help="Artifact name to use for eval data")
    args = parser.parse_args()

    # Create WanDuckDb instance
    duck = WanDuckDb(artifact_name=args.artifact)

    # Create a SQL query that filters the data based on policies, eval types, etc.
    # For example, assume our JSON has columns like "policy_name", "eval_name", and the metric column.
    policies = "', '".join(args.policies.split(","))
    evals = "', '".join(args.evals.split(","))
    sql = f"""
    SELECT policy_name,
           avg("{args.metric}") as avg_metric,
           count(*) as num_matches
    FROM eval_data
    WHERE policy_name IN ('{policies}')
      AND eval_name IN ('{evals}')
    GROUP BY policy_name
    ORDER BY avg_metric DESC
    """
    # Run the query and print the results
    df = duck.query(sql)
    print(df)

if __name__ == "__main__":
    main()
