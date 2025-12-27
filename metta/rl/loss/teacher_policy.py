from __future__ import annotations

import contextlib
import os
from typing import Any

import torch

from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.mpt_policy import MptPolicy
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
    suppress_nim_stdout = (
        "nim_agents" in teacher_spec.class_path
        and os.getenv("METTA_SUPPRESS_NIM_UNKNOWN_FEATURES", "1") != "0"
    )
    with _suppress_stdout_stderr(enabled=suppress_nim_stdout):
        teacher_policy = initialize_or_load_policy(policy_env_info, teacher_spec)
    if unwrap_mpt and isinstance(teacher_policy, MptPolicy):
        teacher_policy = teacher_policy._policy
    return teacher_policy


@contextlib.contextmanager
def _suppress_stdout_stderr(*, enabled: bool) -> None:
    if not enabled:
        yield
        return
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)
