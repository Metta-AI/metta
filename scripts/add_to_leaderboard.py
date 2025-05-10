import argparse
import subprocess
import sys
from typing import List


def get_db_uri(simulation_suite: str) -> str:
    return f"wandb://artifacts/{simulation_suite}_db"


def get_dashboard_path(dashboard_name: str) -> str:
    return f"s3://softmax-public/policydash/{dashboard_name}.html"


def get_dashboard_url(dashboard_name: str) -> str:
    return f"https://softmax-public.s3.amazonaws.com/policydash/{dashboard_name}.html"


def add_to_leaderboard(
    sim_suite: str, policy_uri: str, dashboard_name_opt: str | None, eval_db_uri_opt: str | None, extra_args: List[str]
):
    run_name = f"{sim_suite}_leaderboard"
    dashboard_name = dashboard_name_opt or f"{sim_suite}_leaderboard"
    eval_db_uri = eval_db_uri_opt or get_db_uri(sim_suite)

    print(f"Adding policy to eval leaderboard with policy URI: {policy_uri}")
    print(f"Simulation suite: {sim_suite}")
    print(f"Eval DB URI: {eval_db_uri}")

    # Step 1: Run the simulation
    # TODO: Replace with a normal function call
    print("Step 1: Running simulation...")
    sim_cmd = [
        sys.executable,
        "-m",
        "tools.sim",
        f"sim={sim_suite}",
        f"run={run_name}",
        f"policy_uri={policy_uri}",
        f"+eval_db_uri={eval_db_uri}",
    ] + extra_args

    print(f"Executing: {' '.join(sim_cmd)}")
    sim_proc = subprocess.run(sim_cmd)
    if sim_proc.returncode != 0:
        print("Error: Simulation failed. Exiting.")
        sys.exit(1)

    # Step 3: Analyze and update dashboard
    print("Step 2: Analyzing results and updating dashboard...")
    analyze_cmd = [
        sys.executable,
        "-m",
        "tools.analyze",
        f"run={run_name}",
        f"+eval_db_uri={eval_db_uri}",
        f"analyzer.output_path={get_dashboard_path(dashboard_name)}",
        "+analyzer.num_output_policies=all",
    ] + extra_args

    print(f"Executing: {' '.join(analyze_cmd)}")
    analyze_proc = subprocess.run(analyze_cmd)
    if analyze_proc.returncode != 0:
        print("Error: Analysis failed. Exiting.")
        sys.exit(1)

    print("Successfully added policy to leaderboard and updated dashboard!")
    print(f"Dashboard URL: {get_dashboard_url(dashboard_name)}")


SIMULATION_SUITES = ["navigation", "multiagent", "memory"]


def main():
    parser = argparse.ArgumentParser(description="Add a policy to the leaderboard.")
    parser.add_argument("--policy_uri", type=str, required=True, help="Policy URI (e.g., wandb://run/b.user.test_run)")
    parser.add_argument(
        "--sim_suite",
        type=str,
        required=True,
        choices=SIMULATION_SUITES,
        help="Simulation suite (navigation, multiagent, memory)",
    )
    parser.add_argument("--eval_db_uri", type=str, required=False, help="Evaluation DB URI (optional)")
    parser.add_argument("--dashboard_name", type=str, required=False, help="Dashboard name (optional)")
    args, extra_args = parser.parse_known_args()

    add_to_leaderboard(args.sim_suite, args.policy_uri, args.dashboard_name, args.eval_db_uri, extra_args)


if __name__ == "__main__":
    main()
