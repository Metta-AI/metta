import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, DefaultDict, Iterable, Optional, Sequence

import typer
from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, TextColumn

import gitta as git
from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success


@dataclass
class FormatterResult:
    success: bool
    output: str = ""
    processed_files: int = 0


class FormatterConfig(BaseModel):
    name: str
    format_cmds: tuple[tuple[str, ...], ...] = ()
    check_cmds: tuple[tuple[str, ...], ...] = ()
    extensions: tuple[str, ...] = ()
    runner: Callable[[bool, set[str] | None], FormatterResult] | None = None
    accepts_file_args: bool = True

    def run(self, fix: bool = False, files: set[str] | None = None) -> FormatterResult:
        if self.runner is not None:
            return self.runner(fix, files)
        commands = self.check_cmds if not fix else self.format_cmds
        if not commands:
            return FormatterResult(success=True, processed_files=len(files or []))
        file_count = len(files or [])
        file_args = sorted(files) if files else []
        if file_args and not self.accepts_file_args:
            file_args = []
        for base_cmd in commands:
            cmd = list(base_cmd)
            if file_args:
                cmd.extend(file_args)
            result = subprocess.run(
                cmd,
                cwd=get_repo_root(),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return FormatterResult(
                    success=False,
                    output=result.stderr + result.stdout,
                    processed_files=file_count,
                )
        return FormatterResult(success=True, processed_files=file_count)


def _normalize_extensions(extensions: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for ext in extensions:
        normalized.append(ext if ext.startswith(".") else f".{ext}")
    return tuple(normalized)


def _resolve_target_files(files: Optional[Sequence[str]], staged: bool) -> list[str]:
    repo_root = get_repo_root()
    normalized: list[str] = []
    seen: set[str] = set()
    if files is not None:
        raw_files = list(files)
    elif staged:
        raw_files = [
            fname for status, fname in git.get_uncommitted_files_and_statuses() if status[0] in ("M", "A", "R")
        ]
    else:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        raw_files = result.stdout.splitlines() if result.returncode == 0 and result.stdout else []

    for raw in raw_files:
        if not raw:
            continue
        path = Path(raw)
        if path.is_absolute():
            try:
                path = path.relative_to(repo_root)
            except ValueError:
                path = path.resolve()
        rel = path.as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        if not (repo_root / rel).exists():
            continue
        normalized.append(rel)
    return normalized


def _collect_prettier_targets(
    extensions: Sequence[str],
    files: set[str] | None,
    exclude_patterns: Sequence[str],
) -> list[str]:
    normalized_exts = _normalize_extensions(extensions)
    repo_root = get_repo_root()
    if not files:
        return []
    candidates = []
    for candidate in sorted(files):
        if Path(candidate).suffix.lower() not in normalized_exts:
            continue
        if not (repo_root / candidate).exists():
            continue
        if exclude_patterns and any(pattern in candidate for pattern in exclude_patterns):
            continue
        candidates.append(candidate)
    return candidates


def _format_progress_message(formatter_name: str, action: str, file_count: int, color: str) -> str:
    return f"[{color}]{formatter_name} • {action} {file_count} file(s)[/]"


def _make_cpp_runner() -> Callable[[bool, set[str] | None], FormatterResult]:
    """Create a runner for C++ formatting using clang-format."""

    def _runner(fix: bool, _files: set[str] | None) -> FormatterResult:
        repo_root = get_repo_root()
        mettagrid_dir = repo_root / "packages" / "mettagrid"

        # Find all C/C++ files in the target directories
        cpp_extensions = (".c", ".h", ".cpp", ".hpp")
        target_dirs = ["cpp/src", "cpp/include", "tests", "benchmarks"]
        cpp_files: list[str] = []

        for target_dir in target_dirs:
            dir_path = mettagrid_dir / target_dir
            if not dir_path.exists():
                continue
            for ext in cpp_extensions:
                cpp_files.extend(str(f) for f in dir_path.rglob(f"*{ext}"))

        if not cpp_files:
            return FormatterResult(success=True, processed_files=0)

        if fix:
            cmd = ["clang-format", "-i", "-style=file", *cpp_files]
        else:
            cmd = ["clang-format", "--dry-run", "--Werror", "-style=file", *cpp_files]

        result = subprocess.run(
            cmd,
            cwd=mettagrid_dir,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return FormatterResult(
                success=False,
                output=result.stderr + result.stdout,
                processed_files=len(cpp_files),
            )

        return FormatterResult(success=True, processed_files=len(cpp_files))

    return _runner


def _make_prettier_runner(
    *,
    extensions: Sequence[str],
    exclude_patterns: Sequence[str] = (),
) -> Callable[[bool, set[str] | None], FormatterResult]:
    def _runner(fix: bool, files: set[str] | None) -> FormatterResult:
        targets = _collect_prettier_targets(extensions, files, exclude_patterns)

        if not targets:
            return FormatterResult(success=True, processed_files=0)

        def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
            for start in range(0, len(items), size):
                yield items[start : start + size]

        mode_arg = "--write" if fix else "--check"
        processed = len(targets)
        for chunk in _chunked(targets, 100):  # Prettier's default batch size is 100
            cmd = ["pnpm", "exec", "prettier", mode_arg, *chunk]
            result = subprocess.run(
                cmd,
                cwd=get_repo_root(),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return FormatterResult(
                    success=False,
                    output=result.stderr + result.stdout,
                    processed_files=processed,
                )

        return FormatterResult(success=True, processed_files=processed)

    return _runner


def get_formatters() -> dict[str, FormatterConfig]:
    return {
        f.name: f
        for f in [
            FormatterConfig(
                name="Python",
                # --force-exclude makes these commands respect .ruffignore and ruff.toml's exclude section
                format_cmds=(
                    ("uv", "run", "ruff", "check", "--fix", "--force-exclude"),
                    ("uv", "run", "ruff", "format", "--force-exclude"),
                ),
                check_cmds=(
                    ("uv", "run", "ruff", "check", "--force-exclude"),
                    ("uv", "run", "ruff", "format", "--check", "--force-exclude"),
                ),
                extensions=(".py",),
            ),
            FormatterConfig(
                name="Python Import Linter",
                check_cmds=(("uv", "run", "lint-imports"),),
                format_cmds=(("uv", "run", "lint-imports"),),
                extensions=(".py",),
                accepts_file_args=False,
            ),
            FormatterConfig(
                name="JSON",
                extensions=(".json", ".jsonc", ".code-workspace"),
                runner=_make_prettier_runner(
                    extensions=(".json", ".jsonc", ".code-workspace"),
                    exclude_patterns=(
                        "/charts/",
                        "packages/mettagrid/python/src/mettagrid/renderer/assets/",
                        "packages/mettagrid/nim/mettascope/data/",
                        ".import_linter_cache/",
                    ),
                ),
            ),
            FormatterConfig(
                name="Markdown",
                extensions=(".md",),
                runner=_make_prettier_runner(
                    extensions=(".md",),
                ),
            ),
            FormatterConfig(
                name="Shell",
                extensions=(".sh", ".bash"),
                runner=_make_prettier_runner(
                    extensions=(".sh", ".bash"),
                ),
            ),
            FormatterConfig(
                name="TOML",
                extensions=(".toml",),
                runner=_make_prettier_runner(
                    extensions=(".toml",),
                ),
            ),
            FormatterConfig(
                name="YAML",
                extensions=(".yaml", ".yml"),
                runner=_make_prettier_runner(
                    extensions=(".yaml", ".yml"),
                    exclude_patterns=(
                        "/configs/",
                        "/scenes/",
                        "/charts/",
                        ".github/actions/asana/pr_gh_to_asana/test/",
                    ),
                ),
            ),
            FormatterConfig(
                name="C++",
                extensions=(".cpp", ".hpp", ".h", ".c"),
                runner=_make_cpp_runner(),
                accepts_file_args=False,
            ),
            FormatterConfig(
                name="Javascript",
                extensions=(".js", ".jsx", ".ts", ".tsx"),
                runner=_make_prettier_runner(
                    extensions=(".js", ".jsx", ".ts", ".tsx"),
                ),
            ),
        ]
    }


app = typer.Typer(
    help="Code formatters",
    invoke_without_command=True,
)


@app.callback()
def cmd_lint(
    files: Annotated[Optional[list[str]], typer.Argument()] = None,
    staged: Annotated[bool, typer.Option("--staged", help="Only lint staged files")] = False,
    fix: Annotated[bool, typer.Option("--fix", help="Apply fixes automatically")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Print total lint time")] = False,
):
    """Run linting and formatting on code files.

    Examples:
        metta lint                                     # Check all files
        metta lint --fix                               # Autofix all files
        metta lint --staged --fix                      # Autofix staged files
        metta lint path/to/file1 path/to/file2 --fix   # Autofix specific files
    """
    start_time = time.perf_counter()

    # Get available formatters
    formatters = get_formatters()

    # Determine which files to process
    target_files = _resolve_target_files(files, staged)

    files_by_formatter: DefaultDict[str, set[str]] = defaultdict(set)
    for f in target_files or []:
        ext = Path(f).suffix.lower()
        for formatter_name, formatter in formatters.items():
            if ext in formatter.extensions:
                files_by_formatter[formatter_name].add(f)

    # Run formatters for each type
    if not files_by_formatter:
        info("No matching files to lint.")
        if verbose:
            elapsed = time.perf_counter() - start_time
            info(f"Lint finished in {elapsed:.2f}s")
        return

    failed_formatters: list[tuple[str, str]] = []
    columns = (
        SpinnerColumn(style="cyan"),
        TextColumn("{task.description}", justify="left"),
    )

    formatter_order = [name for name in formatters if name in files_by_formatter]
    with Progress(*columns, transient=True) as progress:
        with ThreadPoolExecutor(max_workers=len(formatter_order) or 1) as executor:
            futures = {}
            for formatter_name in formatter_order:
                formatter = formatters[formatter_name]
                fs = files_by_formatter.get(formatter_name)
                if not fs:
                    continue
                action_word = "Formatting" if fix else "Checking"
                file_count = len(fs)
                in_progress_desc = _format_progress_message(formatter.name, action_word, file_count, "blue")
                task_id = progress.add_task(in_progress_desc, total=None, start=True)
                future = executor.submit(formatter.run, fix=fix, files=fs)
                futures[future] = (formatter, task_id, file_count)

            for future in as_completed(futures):
                formatter, task_id, planned_count = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive guardrail
                    result = FormatterResult(success=False, output=str(exc), processed_files=0)
                processed = result.processed_files or planned_count
                final_action = "Formatted" if fix else "Checked"
                final_color = "green" if result.success else "red"
                final_desc = _format_progress_message(formatter.name, final_action, processed, final_color)
                progress.update(task_id, description=final_desc)
                progress.stop_task(task_id)
                if not result.success:
                    failed_formatters.append((formatter.name, result.output))

    # Print summary
    if failed_formatters:
        error(f"Linting/formatting failed for: {', '.join(name for name, _ in failed_formatters)}")
        for formatter_name, output in failed_formatters:
            if output:
                typer.echo(f"\n[{formatter_name} output]\n{output}\n")
        raise typer.Exit(1)
    else:
        success("All formatting complete")

    if verbose:
        elapsed = time.perf_counter() - start_time
        info(f"Lint finished in {elapsed:.2f}s")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
