"""Policy serialization helpers for CoGames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

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
    if hasattr(policy, "save_checkpoint"):
        policy.save_checkpoint(str(weights_path))
    else:
        torch.save(policy.state_dict(), weights_path)
    metadata = {"policy_class": policy_class_path, "weights": WEIGHTS_NAME}
    (output_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2))
    return PolicyArtifact(policy_class=policy_class_path, weights_path=weights_path)


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


def load_policy(artifact: PolicyArtifact, env: MettaGridEnv, device: torch.device) -> Any:
    policy_class = load_symbol(artifact.policy_class)
    policy = policy_class(env, device)
    if artifact.weights_path and artifact.weights_path.exists():
        if hasattr(policy, "load_checkpoint"):
            policy.load_checkpoint(str(artifact.weights_path))
        else:
            policy.load_state_dict(torch.load(artifact.weights_path, map_location=device))
    return policy
