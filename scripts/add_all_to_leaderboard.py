import argparse

from .add_to_leaderboard import SIMULATION_SUITES, add_to_leaderboard


def main():
    parser = argparse.ArgumentParser(description="Add a policy to the leaderboard.")
    parser.add_argument(
        "--policy_uri_file",
        type=str,
        required=True,
        help="File containing policy URIs, one per line (e.g., wandb://run/b.user.test_run)",
    )
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

    with open(args.policy_uri_file, "r") as f:
        policy_uris = f.readlines()

    for policy_uri in policy_uris:
        stripped_uri = policy_uri.strip()
        if len(stripped_uri) > 0:
            add_to_leaderboard(args.sim_suite, stripped_uri, args.dashboard_name, args.eval_db_uri, extra_args)


if __name__ == "__main__":
    main()
