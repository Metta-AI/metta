import argparse
from typing import List

import wandb
from wandb.errors import CommError


def delete_init_runs(sweep_id: str, entity: str, project: str) -> List[str]:
    deleted_runs = []
    api = wandb.Api()

    try:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    except CommError:
        print(f"Sweep not found: {sweep_id}")
        return deleted_runs

    for run in sweep.runs:
        if run.name.endswith(".init"):
            try:
                run.delete()
                deleted_runs.append(run.name)
                print(f"Deleted run: {run.name}")
            except Exception as e:
                print(f"Error deleting run {run.name}: {e}")

    return deleted_runs


def main():
    parser = argparse.ArgumentParser(description="Delete .init runs from a sweep using wandb API")
    parser.add_argument("--sweep", type=str, required=True, help="Sweep ID")
    parser.add_argument("--entity", type=str, default="metta-research", help="W&B entity")
    parser.add_argument("--project", type=str, default="metta", help="W&B project")
    args = parser.parse_args()

    deleted_runs = delete_init_runs(args.sweep, args.entity, args.project)

    if deleted_runs:
        print(f"Successfully deleted {len(deleted_runs)} .init runs")
    else:
        print("No .init runs found to delete")


if __name__ == "__main__":
    main()
