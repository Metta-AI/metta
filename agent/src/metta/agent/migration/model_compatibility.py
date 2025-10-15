#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import yaml

from metta.common.util.fs import get_repo_root
from metta.rl.training import EnvironmentMetaData

REPO_ROOT: Path = get_repo_root()


def run_git(args: Sequence[str]) -> str:
    """Run a git command and return stdout."""
    completed = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return completed.stdout.strip()


def find_merge_base(base_refs: Sequence[str]) -> Tuple[str, str]:
    """Return the merge-base SHA and the ref it succeeded against."""
    for ref in base_refs:
        try:
            base_sha = run_git(["merge-base", "HEAD", ref])
        except subprocess.CalledProcessError:
            continue
        if base_sha:
            return base_sha, ref
    raise RuntimeError("Unable to determine merge base. Provide one or more valid refs with --base-ref.")


def git_diff(base_sha: str, pathspec: str) -> str:
    """Return the diff between base_sha and the current workspace for pathspec."""
    completed = subprocess.run(
        ["git", "diff", base_sha, "--", pathspec],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return completed.stdout


def _serialize_error(exc: Exception) -> Dict[str, object]:
    error: Dict[str, object] = {
        "message": str(exc),
        "exception_type": exc.__class__.__name__,
        "traceback": traceback.format_exc(),
    }
    if isinstance(exc, subprocess.CalledProcessError):
        error["returncode"] = exc.returncode
        error["cmd"] = [str(item) for item in exc.cmd] if exc.cmd else None
        if exc.stdout:
            error["stdout"] = exc.stdout
        if exc.stderr:
            error["stderr"] = exc.stderr
    return error


def _serialize_env_metadata(env: EnvironmentMetaData) -> Dict[str, object]:
    features: Dict[str, Dict[str, object]] = {}
    for name, feature in env.obs_features.items():
        feature_entry: Dict[str, object] = {}
        if hasattr(feature, "id"):
            feature_entry["id"] = int(feature.id)
        if hasattr(feature, "normalization"):
            feature_entry["normalization"] = float(feature.normalization)
        if hasattr(feature, "dtype"):
            feature_entry["dtype"] = str(feature.dtype)
        if hasattr(feature, "shape"):
            try:
                feature_entry["shape"] = list(feature.shape)
            except TypeError:
                pass
        features[name] = feature_entry

    return {
        "obs_width": env.obs_width,
        "obs_height": env.obs_height,
        "num_agents": env.num_agents,
        "action_names": list(env.action_names),
        "features": features,
        "feature_normalizations": {int(k): float(v) for k, v in env.feature_normalizations.items()},
        "observation_space": repr(env.observation_space),
        "action_space": repr(env.action_space),
    }


@dataclass
class AgentCodebase:
    """Create a structured view of diff data that can be serialized as YAML."""

    base_refs: Sequence[str]
    path: str
    env_metadata: Optional[EnvironmentMetaData] = None

    def to_dict(self) -> Dict[str, str]:
        """Return a dictionary containing merge-base info and path diff."""
        base_sha, used_ref = find_merge_base(self.base_refs)
        diff_text = git_diff(base_sha, self.path)
        return {
            "base_ref": used_ref,
            "merge_base_sha": base_sha,
            "path": self.path,
            "diff": diff_text,
        }

    def build_payload(self) -> Dict[str, object]:
        try:
            report = self.to_dict()
        except Exception as exc:  # pragma: no cover - defensive guard
            return {"status": "error", "error": _serialize_error(exc)}
        payload: Dict[str, object] = {"status": "ready", "report": report}
        if self.env_metadata is not None:
            payload["environment_metadata"] = _serialize_env_metadata(self.env_metadata)
        return payload

    def render_yaml(self) -> bytes:
        payload = self.build_payload()
        return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")

    def extra_files(self) -> Dict[str, bytes]:
        """Return archive-ready files for inclusion in checkpoints."""
        return {"agent_codebase.yaml": self.render_yaml()}

    def start_async(self) -> "AgentCodebaseFuture":
        """Kick off background generation of the report."""
        return AgentCodebaseFuture(self)


class AgentCodebaseFuture:
    """Background computation helper for model compatibility metadata."""

    def __init__(self, report: AgentCodebase) -> None:
        self._report = report
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._files: Dict[str, bytes] | None = None

        thread = threading.Thread(
            target=self._run,
            name="model_compatibility_report",
            daemon=True,
        )
        thread.start()

    def _run(self) -> None:
        try:
            files = self._report.extra_files()
        except Exception as exc:  # pragma: no cover - defensive guard
            files = {
                "agent_codebase.yaml": yaml.safe_dump(
                    {"status": "error", "error": _serialize_error(exc)},
                    sort_keys=False,
                ).encode("utf-8")
            }
        with self._lock:
            self._files = files
        self._ready.set()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def files(self) -> Dict[str, bytes]:
        if not self._ready.is_set():
            return {}
        with self._lock:
            if not self._files:
                return {}
            return dict(self._files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Show the merge-base SHA with main and the diff against the current workspace restricted to a path."
        )
    )
    parser.add_argument(
        "--base-ref",
        action="append",
        help=(
            "Reference to compare against. "
            "Can be given multiple times to specify fallbacks. "
            "Defaults to origin/main then main."
        ),
    )
    parser.add_argument(
        "--path",
        default="agent/src/metta/agent",
        help="Path to include in the diff (default: agent/src/metta/agent).",
    )
    args = parser.parse_args(argv)

    base_refs = args.base_ref or ["origin/main", "main"]

    report = AgentCodebase(base_refs=base_refs, path=args.path)

    try:
        result = report.to_dict()
    except RuntimeError as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        parser.error(
            f"git command failed with exit code {exc.returncode}: {exc.stderr.strip() if exc.stderr else 'no stderr'}"
        )

    print(f"Merge base against {result['base_ref']}: {result['merge_base_sha']}")

    diff_text = result["diff"]
    if diff_text:
        print("--- diff start ---")
        print(diff_text.rstrip())
        print("--- diff end ---")
    else:
        print("No differences detected for the specified path.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
