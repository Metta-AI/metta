import os
import sys
from typing import Any


def parse_config() -> dict[str, Any]:
    """Parse configuration from environment variables."""
    config = {
        "github_token": os.environ.get("INPUT_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN"),
        "workflow_name": os.environ.get("INPUT_WORKFLOW_NAME"),
        "artifact_name_pattern": os.environ.get("INPUT_ARTIFACT_NAME_PATTERN"),
        "num_artifacts": int(os.environ.get("INPUT_NUM_ARTIFACTS", "5")),
        "output_directory": os.environ.get("INPUT_OUTPUT_DIRECTORY", "downloaded-artifacts"),
        "repo": os.environ.get("GITHUB_REPOSITORY"),
    }

    # Validation
    required_fields = ["github_token", "workflow_name", "artifact_name_pattern", "repo"]
    missing_fields = [field for field in required_fields if not config[field]]

    if missing_fields:
        print(f"❌ Missing required configuration: {', '.join(missing_fields)}")
        sys.exit(1)

    if config["num_artifacts"] < 1 or config["num_artifacts"] > 100:
        print("❌ num_artifacts must be between 1 and 100")
        sys.exit(1)

    return config
