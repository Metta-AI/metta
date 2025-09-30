"""Policy serialization helpers for CoGames."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from cogames.policy import TrainablePolicy
from mettagrid import MettaGridEnv
from mettagrid.util.module import load_symbol

METADATA_NAME = "policy.json"
WEIGHTS_NAME = "policy.pt"


@dataclass
class PolicyArtifact:
    policy_class: str
    weights_path: Optional[Path]


def save_policy(policy: Any, output_dir: Path, *, class_path: Optional[str] = None) -> PolicyArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_class_path = class_path or f"{policy.__class__.__module__}.{policy.__class__.__qualname__}"
    weights_path = output_dir / WEIGHTS_NAME
    saved_weights = False

    if isinstance(policy, TrainablePolicy):
        policy.save_policy_data(str(weights_path))
        saved_weights = True
    elif hasattr(policy, "state_dict"):
        torch.save(policy.state_dict(), weights_path)
        saved_weights = True

    metadata = {"policy_class": policy_class_path}
    if saved_weights:
        metadata["weights"] = WEIGHTS_NAME
    (output_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2))

    return PolicyArtifact(
        policy_class=policy_class_path,
        weights_path=weights_path if saved_weights else None,
    )


def bundle_policy(class_path: str, checkpoint_path: Path, destination: Path) -> PolicyArtifact:
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / WEIGHTS_NAME
    weights_path.write_bytes(checkpoint_path.read_bytes())
    metadata = {"policy_class": class_path, "weights": WEIGHTS_NAME}
    (destination / METADATA_NAME).write_text(json.dumps(metadata, indent=2))
    return PolicyArtifact(policy_class=class_path, weights_path=weights_path)


def load_policy_from_bundle(bundle_dir: Path, env: MettaGridEnv, device: torch.device) -> Any:
    metadata = json.loads((bundle_dir / METADATA_NAME).read_text())
    weights_rel = metadata.get("weights", WEIGHTS_NAME)
    artifact = PolicyArtifact(policy_class=metadata["policy_class"], weights_path=bundle_dir / weights_rel)
    return load_policy(artifact, env, device)


def inspect_bundle(bundle_dir: Path) -> dict[str, Any]:
    metadata_path = bundle_dir / METADATA_NAME
    metadata = json.loads(metadata_path.read_text())
    metadata["weights_path"] = str((bundle_dir / metadata.get("weights", WEIGHTS_NAME)).resolve())
    return metadata


def load_policy(artifact: PolicyArtifact, env: MettaGridEnv, device: torch.device) -> Any:
    policy_class = load_symbol(artifact.policy_class)
    policy = policy_class(env, device)
    if artifact.weights_path and artifact.weights_path.exists():
        if isinstance(policy, TrainablePolicy):
            policy.load_policy_data(str(artifact.weights_path))
        elif hasattr(policy, "load_state_dict"):
            state_dict = torch.load(artifact.weights_path, map_location=device)
            policy.load_state_dict(state_dict)
        else:
            raise ValueError(f"Policy {artifact.policy_class} does not support loading weights")
    elif isinstance(policy, TrainablePolicy):
        LOGGER.warning(
            "No weights provided for trainable policy %s; using random initialization.",
            artifact.policy_class,
        )
    return policy


LOGGER = logging.getLogger(__name__)
