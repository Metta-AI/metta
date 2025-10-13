#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from metta.agent.migration.checkpoint_compatibility import check_checkpoint_compatibility
from metta.rl.policy_artifact import load_policy_artifact


def _format_query(report_dict: dict[str, Any]) -> str:
    """
    Build an LLM-friendly prompt describing the incompatibilities.

    The prompt assumes the assistant (Codex) has access to the repository and can
    generate migration patches.
    """
    checkpoint = report_dict["checkpoint_path"]
    details = json.dumps(report_dict, indent=2, sort_keys=False)
    return (
        "You are Codex with full knowledge of the Metta repository.\n"
        "A checkpoint created by older code failed compatibility checks against the current codebase.\n"
        f"Checkpoint path: {checkpoint}\n"
        "Compatibility report (JSON):\n"
        f"{details}\n\n"
        "Please produce a Python migration script (or code changes) that will load the old checkpoint\n"
        "and transform it so that it is compatible with the current architecture. Include explanations\n"
        "for the transformations you apply."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run agent checkpoint compatibility analysis on a .mpt file."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the checkpoint (.mpt) file to analyse.",
    )
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint file not found: {checkpoint_path}")

    try:
        artifact = load_policy_artifact(checkpoint_path)
    except Exception as exc:  # pragma: no cover - defensive guard
        parser.error(f"Unable to load checkpoint: {exc}")

    if artifact.policy_architecture is None:
        parser.error(
            "Checkpoint does not contain a policy architecture manifest. "
            "Compatibility analysis requires an embedded architecture."
        )

    report = check_checkpoint_compatibility(
        checkpoint_path,
        policy_architecture=artifact.policy_architecture,
    )

    if report.success:
        print(f"SUCCESS: Checkpoint '{checkpoint_path}' is compatible with current code.")
        return 0

    report_dict = report.to_dict()
    query = _format_query(report_dict)

    print("QUERY:")
    print(query)
    return 1


if __name__ == "__main__":
    sys.exit(main())
