from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TEST_TARGETS: tuple[str, ...] = (
    "tests",
    "mettascope/tests",
    "agent/tests",
    "app_backend/tests",
    "common/tests",
    "packages/codebot/tests",
    "packages/cogames/tests",
    "packages/gitta/tests",
    "packages/mettagrid/tests",
)


@dataclass(frozen=True)
class Suite:
    name: str
    relative_cwd: str
    coverage_module: str
    env_gate: str | None = None
    allow_missing: bool = False

    @property
    def cwd(self) -> Path:
        return REPO_ROOT / self.relative_cwd


SUITES: tuple[Suite, ...] = (
    Suite(name="mettagrid", relative_cwd="packages/mettagrid", coverage_module="mettagrid"),
    Suite(name="agent", relative_cwd="agent", coverage_module="agent"),
    Suite(name="common", relative_cwd="common", coverage_module="common"),
    Suite(
        name="app_backend", relative_cwd="app_backend", coverage_module="app_backend", env_gate="RUN_APP_BACKEND_TESTS"
    ),
    Suite(name="codebot", relative_cwd="packages/codebot", coverage_module="codebot"),
    Suite(name="cogames", relative_cwd="packages/cogames", coverage_module="cogames"),
    Suite(name="cortex", relative_cwd="packages/cortex", coverage_module="cortex", allow_missing=True),
    Suite(name="gitta", relative_cwd="packages/gitta", coverage_module="gitta", allow_missing=True),
    Suite(name="core", relative_cwd="core", coverage_module="core", allow_missing=True),
)


def _env_str_to_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    return lowered not in {"0", "false", "no", "off"} if default else lowered in {"1", "true", "yes", "on"}


def _should_skip_suite(suite: Suite, *, explicit_skip_app_backend: bool | None) -> bool:
    if suite.name == "app_backend":
        if explicit_skip_app_backend is True:
            return True
        if explicit_skip_app_backend is False:
            return False
        return not _env_str_to_bool(os.environ.get(suite.env_gate or ""), default=True)
    if suite.env_gate:
        return not _env_str_to_bool(os.environ.get(suite.env_gate), default=True)
    return False


def _run_command(args: Sequence[str], *, cwd: Path) -> int:
    print(f"→ {' '.join(args)} (cwd={cwd})")
    completed = subprocess.run(args, cwd=cwd, check=False)
    return completed.returncode


def _run_default(extra_pytest_args: Sequence[str]) -> int:
    workers = os.environ.get("METTA_TEST_WORKERS", "auto")
    cmd = [
        "uv",
        "run",
        "pytest",
        *DEFAULT_TEST_TARGETS,
        "--benchmark-disable",
        "-n",
        workers,
        *extra_pytest_args,
    ]
    return _run_command(cmd, cwd=REPO_ROOT)


def _run_ci(
    *,
    coverage_dir: Path,
    extra_pytest_args: Sequence[str],
    explicit_skip_app_backend: bool | None,
) -> int:
    coverage_dir.mkdir(parents=True, exist_ok=True)
    base_args = [
        "-n",
        os.environ.get("METTA_TEST_WORKERS", "4"),
        "--timeout=100",
        "--timeout-method=thread",
        "--cov-branch",
        "--benchmark-skip",
        "--maxfail=1",
        "--disable-warnings",
        "--durations=10",
        "-v",
    ]

    overall_exit = 0
    for suite in SUITES:
        if _should_skip_suite(suite, explicit_skip_app_backend=explicit_skip_app_backend):
            print(f"↷ Skipping {suite.name} suite")
            continue

        if not suite.cwd.exists():
            if suite.allow_missing:
                print(f"↷ Skipping {suite.name} suite (directory missing)")
                continue
            print(f"✖ Suite directory missing: {suite.cwd}", file=sys.stderr)
            return 1

        coverage_target = coverage_dir / f"coverage-{suite.name}.xml"
        coverage_target.parent.mkdir(parents=True, exist_ok=True)
        coverage_arg = os.path.relpath(coverage_target, start=suite.cwd)

        cmd = [
            "uv",
            "run",
            "pytest",
            *base_args,
            f"--cov={suite.coverage_module}",
            f"--cov-report=xml:{coverage_arg}",
            *extra_pytest_args,
        ]

        exit_code = _run_command(cmd, cwd=suite.cwd)
        if exit_code != 0:
            return exit_code
        overall_exit = max(overall_exit, exit_code)

    return overall_exit


def run_preset(
    preset: str,
    *,
    extra_pytest_args: Sequence[str],
    coverage_dir: Path | None = None,
    skip_app_backend: bool | None = None,
) -> int:
    normalized = preset.lower()
    if normalized == "default":
        return _run_default(extra_pytest_args)
    if normalized in {"ci", "coverage"}:
        resolved_dir = coverage_dir if coverage_dir is not None else (REPO_ROOT / "coverage-reports")
        if not resolved_dir.is_absolute():
            resolved_dir = REPO_ROOT / resolved_dir
        return _run_ci(
            coverage_dir=resolved_dir,
            extra_pytest_args=extra_pytest_args,
            explicit_skip_app_backend=skip_app_backend,
        )
    raise ValueError(f"Unknown preset '{preset}'")


def available_presets() -> tuple[str, ...]:
    return ("default", "ci")


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Metta test presets from a unified entry point.")
    parser.add_argument(
        "--preset",
        "-p",
        default="default",
        choices=available_presets(),
        help="Test preset to run.",
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        default=Path("coverage-reports"),
        help="Directory to store coverage XML files (ci preset only).",
    )
    parser.add_argument(
        "--skip-app-backend",
        action="store_true",
        help="Skip the app_backend test suite even if enabled via environment.",
    )

    parsed, extra = parser.parse_known_args(argv)
    return parsed, extra


def main(argv: Sequence[str] | None = None) -> int:
    args, extra_pytest_args = parse_args(argv)
    return run_preset(
        args.preset,
        extra_pytest_args=extra_pytest_args,
        coverage_dir=args.coverage_dir,
        skip_app_backend=True if args.skip_app_backend else None,
    )


if __name__ == "__main__":
    sys.exit(main())
