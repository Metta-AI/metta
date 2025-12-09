#!/usr/bin/env python3
"""
Placeholder script to generate eval_health.json.

TODO: This is a temporary placeholder. Once Nishad finalizes exact eval data paths,
      implement the following:
      1. Discover eval runs (from S3, database, or API)
      2. Parse eval logs/results for:
         - Success/failure
         - heart_delta_pct (from eval output)
         - Duration (from timestamps or logs)
      3. Differentiate local vs remote evals
      4. Write to output path

Expected output format:
{
  "local": {"success": 1, "heart_delta_pct": 0.1},
  "remote": {"success": 1, "heart_delta_pct": 0.05, "duration_minutes": 42}
}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(os.environ.get("EVAL_SUMMARY_FILE", "/app/devops/datadog/data/eval_health.json"))


def main() -> None:
    logger.warning(
        "generate_eval_health.py is a PLACEHOLDER. "
        "It writes minimal JSON structure. Eval log parsing will be implemented once data paths are finalized."
    )

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write minimal expected JSON structure
    # TODO: Replace with actual eval log parsing logic
    placeholder_data = {
        "local": {
            "success": 0,  # Placeholder - will be 1 when actual data is parsed
            "heart_delta_pct": 0.0,  # Placeholder
        },
        "remote": {
            "success": 0,  # Placeholder
            "heart_delta_pct": 0.0,  # Placeholder
            "duration_minutes": 0,  # Placeholder
        },
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(placeholder_data, fp, indent=2)
        fp.write("\n")

    logger.info("Wrote placeholder eval_health.json to %s", OUTPUT_PATH)
    logger.info("TODO: Implement eval log parsing to populate real values")


if __name__ == "__main__":
    main()
