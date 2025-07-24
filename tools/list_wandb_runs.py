#!/usr/bin/env python3
"""List available WandB runs that can be used as NPCs for dual-policy training."""

import argparse
import sys
from typing import List

import wandb


def list_wandb_runs(
    entity: str = "metta-research",
    project: str = "metta",
    limit: int = 20,
    state_filter: str = "finished",
) -> List[dict]:
    """List WandB runs that can be used as NPCs."""
    try:
        api = wandb.Api()
        runs = api.runs(
            f"{entity}/{project}",
            filters={"state": state_filter},
            order="-created_at",
        )

        run_list = []
        for i, run in enumerate(runs):
            if i >= limit:
                break

            # Get the score if available
            score = run.summary.get("score", "N/A")
            if score == "N/A":
                score = run.summary.get("reward", "N/A")

            run_info = {
                "name": run.name,
                "id": run.id,
                "state": run.state,
                "created_at": run.created_at,
                "score": score,
                "wandb_uri": f"wandb://{entity}/{project}/model/{run.name}:latest",
            }
            run_list.append(run_info)

        return run_list

    except Exception as e:
        print(f"Error accessing WandB API: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="List available WandB runs for dual-policy training")
    parser.add_argument("--entity", default="metta-research", help="WandB entity name (default: metta-research)")
    parser.add_argument("--project", default="metta", help="WandB project name (default: metta)")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of runs to list (default: 20)")
    parser.add_argument("--state", default="finished", help="Filter by run state (default: finished)")
    parser.add_argument(
        "--format", choices=["table", "json", "uris"], default="table", help="Output format (default: table)"
    )

    args = parser.parse_args()

    print(f"Fetching {args.limit} {args.state} runs from {args.entity}/{args.project}...")
    print()

    runs = list_wandb_runs(args.entity, args.project, args.limit, args.state)

    if not runs:
        print("No runs found.")
        sys.exit(1)

    if args.format == "table":
        print(f"{'Run Name':<50} {'Score':<10} {'Created':<12} {'WandB URI'}")
        print("-" * 120)
        for run in runs:
            created_date = run["created_at"][:10] if run["created_at"] else "N/A"
            print(f"{run['name']:<50} {str(run['score']):<10} {created_date:<12} {run['wandb_uri']}")

    elif args.format == "json":
        import json

        print(json.dumps(runs, indent=2))

    elif args.format == "uris":
        for run in runs:
            print(run["wandb_uri"])

    print()
    print("To use one of these runs as an NPC, copy the WandB URI and use it in your config:")
    print('  checkpoint_path: "wandb://metta-research/metta/model/run_name:latest"')
    print()
    print("Or use the recipe script:")
    print('  ./recipes/dual_policy_checkpoint.sh "wandb://metta-research/metta/model/run_name:latest"')


if __name__ == "__main__":
    main()
