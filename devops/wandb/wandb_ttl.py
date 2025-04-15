import argparse

import wandb

ENTITY_NAME = "metta-research"
PROJECT_NAME = "metta"


def apply_ttl_to_artifacts(artifact_project: str, version_start: int, version_end: int, ttl_days: int) -> None:
    """
    Update the TTL of a range of artifact versions.
    """
    run = wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME)

    version = version_start
    while version <= version_end:
        try:
            artifact_name = f"{ENTITY_NAME}/{PROJECT_NAME}/{artifact_project}:v{version}"
            artifact = run.use_artifact(artifact_name)
            print(f"Applying TTL to artifact: {artifact_name}")
            artifact.ttl = ttl_days
            artifact.save()
            version += 1
        except wandb.errors.CommError:
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
