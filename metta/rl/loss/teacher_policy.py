from __future__ import annotations

from typing import Any

import torch

from metta.agent.policy import Policy as MettaPolicy
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.util.uri_resolvers.schemes import policy_spec_from_uri


def load_teacher_policy(
    env: Any,
    *,
    policy_uri: str,
    device: torch.device | str,
    error: str = "Environment metadata is required to instantiate teacher policy",
):
    policy_env_info = getattr(env, "policy_env_info", None)
    if policy_env_info is None:
        raise RuntimeError(error)

    teacher_spec = policy_spec_from_uri(policy_uri, device=str(device))
    teacher_policy = initialize_or_load_policy(policy_env_info, teacher_spec, device_override=str(device))
    if not isinstance(teacher_policy, MettaPolicy):
        raise TypeError(f"Teacher policy must be a torch Policy; got {type(teacher_policy).__name__}")
    return teacher_policy
