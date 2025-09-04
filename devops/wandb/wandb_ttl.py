import argparse
from datetime import timedelta

import wandb
from wandb.errors import CommError

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


def apply_ttl_to_artifacts(artifact_project: str, version_start: int, version_end: int, ttl_days: int) -> None:
    """
    Update the TTL of a range of artifact versions.
    """
    run = wandb.init(project=METTA_WANDB_PROJECT, entity=METTA_WANDB_ENTITY)

    version = version_start
    while version <= version_end:
        try:
            artifact_name = f"{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/{artifact_project}:v{version}"
            artifact = run.use_artifact(artifact_name)
            print(f"Applying TTL to artifact: {artifact_name}")
            artifact.ttl = timedelta(days=ttl_days)
            artifact.save()
            version += 1
        except CommError:
            print(f"No more artifacts found after version {version - 1}.")
            break
        except Exception as e:
            print(f"An error occurred with artifact {artifact_name}: {e}")
            version += 1

    print("All artifact versions processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact_project", required=True, help="The project containing the artifacts e.g. p2.train.norm.feat."
    )
    parser.add_argument(
        "--version_start", type=int, required=True, help="The starting version number of the artifacts."
    )
    parser.add_argument("--version_end", type=int, required=True, help="The ending version number of the artifacts.")
    parser.add_argument(
        "--ttl_days",
        type=int,
        required=True,
        help="The TTL in number of days to apply to each artifact. "
        "Note that it's in days from artifact version creation, "
        "not the current date.",
    )

    args = parser.parse_args()

    apply_ttl_to_artifacts(args.artifact_project, args.version_start, args.version_end, args.ttl_days)
