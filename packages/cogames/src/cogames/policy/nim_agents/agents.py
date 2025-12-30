import importlib
import os
import sys
from pathlib import Path
from typing import Sequence

from mettagrid.policy.policy import NimMultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

current_dir = os.path.dirname(os.path.abspath(__file__))


def _looks_like_nim_agents_bindings_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "nim_agents.py").is_file():
        return True
    # Nim compilation outputs a shared lib next to the Python wrapper.
    for candidate in ("libnim_agents.so", "libnim_agents.dylib", "nim_agents.dll"):
        if (path / candidate).is_file():
            return True
    if any(path.glob("libnim_agents.*")):
        return True
    return False


def _candidate_bindings_dirs() -> list[str]:
    """Return plausible generated-bindings directories containing the nim_agents module.

    In some environments (e.g., policy submissions extracted to /tmp/mettagrid-policy-cache),
    the Nim build outputs to "<extraction_root>/bindings/generated" instead of inside the
    installed cogames package tree. We scan sys.path entries for that layout.
    """
    candidates: list[Path] = []

    # Default: build output colocated with this file.
    candidates.append(Path(current_dir) / "bindings" / "generated")

    # Common submission layout: <extraction_root>/bindings/generated where extraction_root is on sys.path.
    for entry in sys.path:
        if not entry:
            continue
        try:
            base = Path(entry).resolve()
        except OSError:
            continue
        candidates.append(base / "bindings" / "generated")

    seen: set[str] = set()
    out: list[str] = []
    for path in candidates:
        if not _looks_like_nim_agents_bindings_dir(path):
            continue
        s = str(path)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _import_nim_agents():
    try:
        return importlib.import_module("nim_agents")
    except ModuleNotFoundError as e:
        # Only handle the case where the top-level module itself is missing.
        if getattr(e, "name", None) != "nim_agents":
            raise

        for bindings_dir in _candidate_bindings_dirs():
            if bindings_dir not in sys.path:
                sys.path.insert(0, bindings_dir)

        try:
            return importlib.import_module("nim_agents")
        except ModuleNotFoundError as e2:
            if getattr(e2, "name", None) != "nim_agents":
                raise
            raise ModuleNotFoundError(
                "No module named 'nim_agents'.\n"
                "Expected generated bindings in one of:\n"
                + "\n".join(f"  - {p}" for p in _candidate_bindings_dirs())
                + "\n\n"
                "If you're running from source, generate bindings by running:\n"
                f"  cd {current_dir} && nim c nim_agents.nim"
            ) from e2


na = _import_nim_agents()


def start_measure():
    na.start_measure()


def end_measure():
    na.end_measure()


class ThinkyAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["thinky"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.ThinkyPolicy,
            agent_ids=agent_ids,
        )


class RandomAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_random"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.RandomPolicy,
            agent_ids=agent_ids,
        )


class RaceCarAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["race_car"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.RaceCarPolicy,
            agent_ids=agent_ids,
        )


class LadyBugAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_ladybug"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.LadybugPolicy,
            agent_ids=agent_ids,
        )
