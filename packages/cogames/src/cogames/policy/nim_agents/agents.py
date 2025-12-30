import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from mettagrid.policy.policy import NimMultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    import fcntl  # type: ignore[import-not-found]  # POSIX-only
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


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


def _all_candidate_bindings_dirs() -> list[Path]:
    """Return plausible generated-bindings directories for the nim_agents module.

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
    out: list[Path] = []
    for path in candidates:
        s = str(path.resolve()) if path.exists() else str(path)
        if s in seen:
            continue
        seen.add(s)
        out.append(path)
    return out


def _existing_candidate_bindings_dirs() -> list[str]:
    out: list[str] = []
    for path in _all_candidate_bindings_dirs():
        if _looks_like_nim_agents_bindings_dir(path):
            out.append(str(path))
    return out


def _ensure_local_bindings_built() -> None:
    """Best-effort build of nim_agents into this package's bindings/generated directory.

    This is needed in runtime environments where cogames is installed from source but the
    Nim bindings are not pre-generated (common on fresh SkyPilot nodes).
    """
    local_dir = Path(current_dir) / "bindings" / "generated"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Fast-path: already generated.
    if (local_dir / "nim_agents.py").is_file():
        return

    lock_path = local_dir / ".nim_agents_build.lock"
    done_marker = local_dir / ".nim_agents_build.done"

    # Another fast-path: a previous process finished successfully.
    if done_marker.is_file() and (local_dir / "nim_agents.py").is_file():
        return

    def _build() -> None:
        # Re-check after acquiring lock.
        if (local_dir / "nim_agents.py").is_file():
            done_marker.write_text("ok\n", encoding="utf-8")
            return

        result = subprocess.run(
            ["nim", "c", "nim_agents.nim"],
            cwd=current_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to build Nim bindings for 'nim_agents'.\n"
                f"Command: nim c nim_agents.nim (cwd: {current_dir})\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
            )
        if not (local_dir / "nim_agents.py").is_file():
            raise RuntimeError(
                "Nim build finished but did not produce bindings/generated/nim_agents.py.\n"
                f"Expected: {local_dir / 'nim_agents.py'}"
            )
        done_marker.write_text("ok\n", encoding="utf-8")

    if fcntl is None:
        _build()
        return

    with open(lock_path, "w", encoding="utf-8") as fp:
        fcntl.flock(fp, fcntl.LOCK_EX)
        _build()


def _import_nim_agents():
    try:
        return importlib.import_module("nim_agents")
    except ModuleNotFoundError as e:
        # Only handle the case where the top-level module itself is missing.
        if getattr(e, "name", None) != "nim_agents":
            raise

        # First, try to build local bindings into this package tree (most reliable).
        try:
            _ensure_local_bindings_built()
        except Exception:
            # If build fails, we still try other candidate dirs (some deployments generate elsewhere).
            pass

        # Add all plausible binding dirs (local + any extracted policy roots on sys.path).
        for bindings_dir in _all_candidate_bindings_dirs():
            b = str(bindings_dir)
            if b not in sys.path:
                sys.path.insert(0, b)

        try:
            return importlib.import_module("nim_agents")
        except ModuleNotFoundError as e2:
            if getattr(e2, "name", None) != "nim_agents":
                raise
            all_candidates = [str(p) for p in _all_candidate_bindings_dirs()]
            existing_candidates = _existing_candidate_bindings_dirs()
            raise ModuleNotFoundError(
                "No module named 'nim_agents'.\n"
                "Expected generated bindings in one of these directories:\n"
                + "\n".join(f"  - {p}" for p in all_candidates)
                + "\n\n"
                "Directories that currently look populated with nim_agents bindings:\n"
                + ("\n".join(f"  - {p}" for p in existing_candidates) if existing_candidates else "  <none>")
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
