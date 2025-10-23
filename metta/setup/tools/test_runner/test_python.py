import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Iterable, Sequence

import typer
from pydantic import BaseModel

from metta.common.util.fs import get_repo_root
from metta.setup.tools.test_runner.junit_summary import (
    emit_annotations,
    merge_reports,
    summarize_report,
)
from metta.setup.utils import error, info


class Package(BaseModel):
    name: str
    target: Path

    @property
    def key(self) -> str:
        return self.name.lower()

    @property
    def target_path(self) -> Path:
        root = get_repo_root()
        if self.target.is_absolute():
            return self.target
        return root / self.target


PACKAGES: tuple[Package, ...] = (
    Package(name="tests", target=Path("tests")),
    Package(name="mettascope", target=Path("mettascope/tests")),
    Package(name="agent", target=Path("agent/tests")),
    Package(name="app_backend", target=Path("app_backend/tests")),
    Package(name="common", target=Path("common/tests")),
    Package(name="codebot", target=Path("packages/codebot/tests")),
    Package(name="cogames", target=Path("packages/cogames/tests")),
    Package(name="gitta", target=Path("packages/gitta/tests")),
    Package(name="mettagrid", target=Path("packages/mettagrid/tests")),
    Package(name="cortex", target=Path("packages/cortex/tests")),
)


DEFAULT_FLAGS: tuple[str, ...] = ("--benchmark-disable", "-n", "auto")
CI_FLAGS: tuple[str, ...] = (
    "-n",
    "4",
    "--timeout=100",
    "--timeout-method=thread",
    "--benchmark-skip",
    "--maxfail=1",
    "--disable-warnings",
    "--durations=10",
    "-v",
)


def _run_command(args: Sequence[str]) -> int:
    info(f"→ {' '.join(args)}")
    completed = subprocess.run(args, cwd=get_repo_root(), check=False)
    return completed.returncode


def _resolve_package_targets(
    include: Iterable[str],
    exclude: Iterable[str],
) -> list[Package]:
    package_map = {package.key: package for package in PACKAGES}
    include_keys = [name.lower() for name in include]
    exclude_keys = {name.lower() for name in exclude}

    selected: list[Package]
    if include_keys:
        missing = [name for name in include_keys if name not in package_map]
        if missing:
            raise ValueError(f"Unknown suite(s): {', '.join(missing)}")
        selected = [package_map[name] for name in include_keys]
    else:
        selected = list(PACKAGES)

    if exclude_keys:
        selected = [package for package in selected if package.key not in exclude_keys]

    return selected


def _collect_package_targets(packages: Iterable[Package]) -> list[str]:
    targets: list[Path] = []
    for package in packages:
        path = package.target_path
        if not path.exists():
            raise ValueError(f"Test package directory missing: {path}")
        targets.append(path)
    if not targets:
        raise ValueError("No package targets resolved.")
    return [str(t.relative_to(get_repo_root())) for t in targets]


app = typer.Typer(
    help="Python test runner",
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)


@app.callback()
def run(
    ctx: typer.Context,
    targets: Annotated[list[str] | None, typer.Argument(help="Explicit pytest targets.")] = None,
    ci: bool = typer.Option(False, "--ci", help="Use CI-style settings and parallel suite execution."),
    packages: Annotated[
        list[str] | None, typer.Option("--package", "-p", help="Limit to specific named package(s).")
    ] = None,
    skip_packages: Annotated[
        list[str] | None, typer.Option("--skip-package", help="Exclude package(s) by name.")
    ] = None,
    changed: bool = typer.Option(
        False,
        "--changed",
        help="Run only tests impacted by staged/changed files using pytest-testmon.",
    ),
    test_summary_out: Annotated[
        Path | None,
        typer.Option(
            "--test-summary-out",
            help="Write a Markdown summary (and companion JUnit XML) to this path. "
            "Provide either a filename or a prefix without extension.",
        ),
    ] = None,
) -> None:
    extra_args = list(ctx.args or [])
    target_args = targets or []
    package_args = packages or []
    skip_package_args = skip_packages or []

    cmd = ["uv", "run", "pytest"]
    if target_args:
        if package_args or skip_package_args or changed or ci:
            error("Explicit targets cannot be combined with suite filters, --changed, or --ci.")
            raise typer.Exit(1)
        summary_path, junit_path = _resolve_summary_paths(test_summary_out)
        if junit_path:
            cmd.extend(["--junitxml", str(junit_path)])
        exit_code = _run_command([*cmd, *target_args, *extra_args])
        _maybe_write_summary(junit_path, summary_path, exit_code)
        raise typer.Exit(exit_code)

    try:
        selected = _resolve_package_targets(package_args, skip_package_args)
        resolved_targets = _collect_package_targets(selected)
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc

    summary_path, junit_path = _resolve_summary_paths(test_summary_out)

    if ci:
        exit_code = _run_ci(resolved_targets, extra_args, junit_path, summary_path)
        raise typer.Exit(exit_code)
    else:
        cmd.extend(DEFAULT_FLAGS)
        if changed:
            cmd.append("--testmon")
        cmd.extend(extra_args)
        if junit_path:
            cmd.extend(["--junitxml", str(junit_path)])
        cmd.extend(resolved_targets)
        exit_code = _run_command(cmd)
        _maybe_write_summary(junit_path, summary_path, exit_code)
        raise typer.Exit(exit_code)


def _resolve_summary_paths(test_summary_out: Path | None) -> tuple[Path | None, Path | None]:
    if test_summary_out is None:
        return None, None

    path = test_summary_out.expanduser()
    if path.suffix:
        summary_path = path
        junit_path = path.with_suffix(".xml")
    else:
        summary_path = path.with_suffix(".md")
        junit_path = path.with_suffix(".xml")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    return summary_path, junit_path


def _maybe_write_summary(junit_path: Path | None, summary_path: Path | None, exit_code: int) -> None:
    if not junit_path or not summary_path:
        return

    if junit_path.exists():
        summary = summarize_report(junit_path)
        summary_path.write_text(summary, encoding="utf-8")
        emit_annotations(junit_path)
        info(f"Test summary written to {summary_path}")
        info(f"JUnit report written to {junit_path}")
    else:
        info("JUnit report not produced; summary skipped.")


def _run_ci(
    targets: Sequence[str],
    extra_args: Sequence[str],
    junit_path: Path | None,
    summary_path: Path | None,
) -> int:
    tmp_dir: Path | None = None
    xml_paths: list[Path] = []
    exit_code = 0

    try:
        if junit_path:
            tmp_dir = Path(tempfile.mkdtemp(prefix="metta-pytest-ci-"))

        def run_target(target: str) -> int:
            cmd = ["uv", "run", "pytest", *CI_FLAGS, *extra_args]
            xml_file: Path | None = None
            if tmp_dir is not None:
                safe_name = target.replace("/", "_").replace(":", "_")
                xml_file = tmp_dir / f"{safe_name}.xml"
                cmd.extend(["--junitxml", str(xml_file)])
            cmd.append(target)
            rc = _run_command(cmd)
            if xml_file and xml_file.exists():
                xml_paths.append(xml_file)
            return rc

        max_workers = min(len(targets), 8) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for code in pool.map(run_target, targets):
                exit_code = max(exit_code, code)

        if junit_path and xml_paths:
            merge_reports(xml_paths, junit_path)
        if summary_path and junit_path and junit_path.exists():
            summary = summarize_report(junit_path)
            summary_path.write_text(summary, encoding="utf-8")
            emit_annotations(junit_path)
            info(f"Test summary written to {summary_path}")
            info(f"JUnit report written to {junit_path}")
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return exit_code
