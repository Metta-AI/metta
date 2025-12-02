import importlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from mettagrid.policy.policy import NimMultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.join(current_dir, "bindings/generated")
if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)


# --------------------------------------------------------------------------- #
# Auto-rebuild Nim bindings when stale or missing.
# This keeps RaceCar/Thinky in sync after Nim edits without manual install.sh.
# --------------------------------------------------------------------------- #
def _ensure_nim_bindings_up_to_date() -> None:
    """Rebuild Nim bindings if any Nim source is newer than the generated lib."""
    if os.environ.get("COGAMES_SKIP_NIM_REBUILD"):
        return

    root = Path(current_dir)
    lib_candidates = list((root / "bindings" / "generated").glob("libnim_agents.*"))
    if not lib_candidates:
        needs_build = True
    else:
        lib_mtime = max(p.stat().st_mtime for p in lib_candidates)
        # Track all Nim sources that affect the bindings.
        nim_sources = list(root.glob("*.nim")) + list(root.glob("*.nims")) + list(root.glob("*.cfg"))
        src_mtime = max(p.stat().st_mtime for p in nim_sources)
        needs_build = src_mtime > lib_mtime + 0.5  # small tolerance

    if not needs_build:
        return

    install_sh = root / "install.sh"
    if not install_sh.exists():
        # Nothing we can do; fall through.
        return
    try:
        subprocess.run(
            ["bash", "-lc", f"cd '{root}' && chmod +x install.sh && ./install.sh"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Small sleep to avoid tight loops on repeated imports.
        time.sleep(0.05)
    except Exception:
        # If build fails, keep going; the import will raise later.
        pass


_ensure_nim_bindings_up_to_date()

na = importlib.import_module("nim_agents")


def start_measure():
    na.start_measure()


def end_measure():
    na.end_measure()


class ThinkyAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_thinky", "thinky"]

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
    short_names = [
        "nim_race_car",
        "racecar",
    ]

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
