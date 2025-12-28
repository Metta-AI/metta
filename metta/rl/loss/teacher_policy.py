from __future__ import annotations

from typing import Any

import torch

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
    teacher_spec = policy_spec_from_uri(policy_uri, device=str(device))
    teacher_policy = initialize_or_load_policy(policy_env_info, teacher_spec, device_override=str(device))
    return teacher_policy
