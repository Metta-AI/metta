from __future__ import annotations

from typing import Any

import torch

from mettagrid.policy.checkpoint_policy import CheckpointPolicy
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri


def load_teacher_policy(
    env: Any,
    *,
    policy_uri: str,
    device: torch.device | str,
    error: str = "Environment metadata is required to instantiate teacher policy",
    unwrap_mpt: bool = True,
):
    policy_env_info = getattr(env, "policy_env_info", None)
    if policy_env_info is None:
        raise RuntimeError(error)

    teacher_spec = policy_spec_from_uri(policy_uri, device=str(device))
    teacher_policy = CheckpointPolicy.from_policy_spec(
        policy_env_info,
        teacher_spec,
        device_override=str(device),
    ).wrapped_policy
    return teacher_policy
