"""Policy serialization helpers for cogames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from mettagrid import MettaGridEnv
from mettagrid.util.module import load_symbol


@dataclass
class PolicyArtifact:
    policy_class: str
    weights_path: Optional[Path]


def save_policy(policy: Any, output_dir: Path, *, class_path: Optional[str] = None) -> PolicyArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_class_path = class_path or f"{policy.__class__.__module__}.{policy.__class__.__qualname__}"
    weights_path = output_dir / "policy.pt"

    if hasattr(policy, "save_checkpoint"):
        policy.save_checkpoint(str(weights_path))
    else:
        torch.save(policy.state_dict(), weights_path)

    return PolicyArtifact(policy_class=policy_class_path, weights_path=weights_path)


def load_policy(artifact: PolicyArtifact, env: MettaGridEnv, device: torch.device) -> Any:
    policy_class = load_symbol(artifact.policy_class)
    policy = policy_class(env, device)

    if artifact.weights_path and artifact.weights_path.exists():
        if hasattr(policy, "load_checkpoint"):
            policy.load_checkpoint(str(artifact.weights_path))
        else:
            policy.load_state_dict(torch.load(artifact.weights_path, map_location=device))

    return policy
