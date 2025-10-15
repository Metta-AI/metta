#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional
from zipfile import ZipFile

from metta.agent.migration.checkpoint_compatibility import check_checkpoint_compatibility
from metta.rl.policy_artifact import load_policy_artifact
from metta.rl.training import EnvironmentMetaData
import yaml


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


def _resolve_env_metadata(func_path: str) -> EnvironmentMetaData:
    if "." not in func_path:
        raise ValueError(f"--env-metadata-func must be a dotted path (module.func): {func_path}")
    module_name, _, attr_name = func_path.rpartition(".")
    module = importlib.import_module(module_name)
    factory: Callable[[], EnvironmentMetaData] = getattr(module, attr_name)
    metadata = factory()
    if not hasattr(metadata, "feature_normalizations"):
        raise TypeError(
            f"Callable '{func_path}' must return an EnvironmentMetaData-like object "
            f"with a 'feature_normalizations' attribute (got {type(metadata)!r})."
        )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run agent checkpoint compatibility analysis on a .mpt file."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the checkpoint (.mpt) file to analyse.",
    )
    parser.add_argument(
        "--env-metadata-func",
        help=(
            "Optional dotted path to a callable that returns an EnvironmentMetaData instance. "
            "Example: experiments.recipes.scratchpad.localmini_env.env_metadata"
        ),
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

    env_metadata: EnvironmentMetaData | None = None
    if args.env_metadata_func:
        env_metadata = _resolve_env_metadata(args.env_metadata_func)
    else:
        env_metadata = _load_env_metadata_from_checkpoint(checkpoint_path)

    report = check_checkpoint_compatibility(
        checkpoint_path,
        policy_architecture=artifact.policy_architecture,
        env_metadata=env_metadata,
    )

    if report.success:
        print(f"SUCCESS: Checkpoint '{checkpoint_path}' is compatible with current code.")
        return 0

    report_dict = report.to_dict()
    query = _format_query(report_dict)

    print("QUERY:")
    print(query)
    return 1


def _load_env_metadata_from_checkpoint(path: Path) -> Optional[EnvironmentMetaData]:
    try:
        with ZipFile(path, "r") as archive:
            if "agent_codebase.yaml" not in archive.namelist():
                return None
            metadata_yaml = archive.read("agent_codebase.yaml").decode("utf-8")
    except Exception:
        return None

    payload = yaml.safe_load(metadata_yaml) or {}
    env_payload = payload.get("environment_metadata")
    if not isinstance(env_payload, dict):
        return None

    feature_norms = {
        int(k): float(v) for k, v in (env_payload.get("feature_normalizations") or {}).items()
    }

    features_payload = env_payload.get("features") or {}
    features = {
        name: SimpleNamespace(**attrs) for name, attrs in features_payload.items()
    }

    try:
        env_metadata = EnvironmentMetaData(
            obs_width=int(env_payload.get("obs_width", 0)),
            obs_height=int(env_payload.get("obs_height", 0)),
            obs_features=features,
            action_names=list(env_payload.get("action_names", [])),
            num_agents=int(env_payload.get("num_agents", 0)),
            observation_space=env_payload.get("observation_space"),
            action_space=env_payload.get("action_space"),
            feature_normalizations=feature_norms,
        )
        return env_metadata
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
