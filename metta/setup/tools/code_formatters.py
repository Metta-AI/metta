import subprocess
from pathlib import Path
from typing import Annotated, Callable, Dict, Iterable, Optional, Sequence, Set

import typer
from pydantic import BaseModel

import gitta as git
from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success


class FormatterConfig(BaseModel):
    name: str
    format_cmds: tuple[tuple[str, ...], ...] = ()
    check_cmds: tuple[tuple[str, ...], ...] = ()
    extensions: tuple[str, ...] = ()
    runner: Callable[[bool, set[str] | None], bool] | None = None
    accepts_file_args: bool = True

    def run(self, fix: bool = False, files: set[str] | None = None) -> bool:
        if self.runner is not None:
            return self.runner(fix, files)
        commands = self.check_cmds if not fix else self.format_cmds
        if not commands:
            return False
        file_args = sorted(files) if files else []
        if file_args and not self.accepts_file_args:
            file_args = []
        for base_cmd in commands:
            cmd = list(base_cmd)
            if file_args:
                cmd.extend(file_args)
            result = subprocess.run(cmd, cwd=get_repo_root(), check=False)
            if result.returncode != 0:
                return False
        return True


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


def _make_prettier_runner(
    *,
    formatter_name: str,
    extensions: Sequence[str],
    exclude_patterns: Sequence[str] = (),
) -> Callable[[bool, set[str] | None], bool]:
    def _runner(fix: bool, files: set[str] | None) -> bool:
        targets = _collect_prettier_targets(extensions, files, exclude_patterns)

        if not targets:
            info(f"{formatter_name}: no matching files to process.")
            return True

        mode_arg = "--write" if fix else "--check"
        action = "Formatting" if fix else "Checking"
        info(f"{formatter_name}: {action.lower()} {len(targets)} files with Prettier...")

        def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
            for start in range(0, len(items), size):
                yield items[start : start + size]

        for chunk in _chunked(targets, 100):  # Prettier's default batch size is 100
            cmd = ["pnpm", "exec", "prettier", mode_arg, *chunk]
            result = subprocess.run(cmd, cwd=get_repo_root(), check=False)
            if result.returncode != 0:
                return False

        return True

    return _runner


def get_formatters() -> dict[str, FormatterConfig]:
    return {
        f.name: f
        for f in [
            FormatterConfig(
                name="Python (ruff)",
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
                name="JSON",
                extensions=(".json", ".jsonc", ".code-workspace"),
                runner=_make_prettier_runner(
                    formatter_name="JSON",
                    extensions=(".json", ".jsonc", ".code-workspace"),
                    exclude_patterns=(
                        "/charts/",
                        "packages/mettagrid/python/src/mettagrid/renderer/assets/",
                        "packages/mettagrid/nim/mettascope/data/",
                    ),
                ),
            ),
            FormatterConfig(
                name="Markdown",
                extensions=(".md",),
                runner=_make_prettier_runner(
                    formatter_name="Markdown",
                    extensions=(".md",),
                ),
            ),
            FormatterConfig(
                name="Shell",
                extensions=(".sh", ".bash"),
                runner=_make_prettier_runner(
                    formatter_name="Shell",
                    extensions=(".sh", ".bash"),
                ),
            ),
            FormatterConfig(
                name="TOML",
                extensions=(".toml",),
                runner=_make_prettier_runner(
                    formatter_name="TOML",
                    extensions=(".toml",),
                ),
            ),
            FormatterConfig(
                name="YAML",
                extensions=(".yaml", ".yml"),
                runner=_make_prettier_runner(
                    formatter_name="YAML",
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
                format_cmds=(("make", "-C", "packages/mettagrid", "format-fix"),),
                check_cmds=(("make", "-C", "packages/mettagrid", "format-check"),),
                extensions=(".cpp", ".hpp", ".h"),
                accepts_file_args=False,
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
):
    """Run linting and formatting on code files.

    Examples:
        metta lint                                     # Check all files
        metta lint --fix                               # Autofix all files
        metta lint --staged --fix                      # Autofix staged files
        metta lint path/to/file1 path/to/file2 --fix   # Autofix specific files
    """
    # Get available formatters
    formatters = get_formatters()

    # Determine which files to process
    target_files = _resolve_target_files(files, staged)

    files_by_formatter: Dict[str, Set[str]] = {}
    for f in target_files or []:
        ext = Path(f).suffix.lower()
        for formatter_name, formatter in formatters.items():
            if formatter.extensions and ext in formatter.extensions:
                files_by_formatter.setdefault(formatter_name, set()).add(f)
                break

    failed_formatters = []

    # Run formatters for each type
    if not files_by_formatter:
        info("No matching files to lint.")
        return

    formatter_order = files_by_formatter.keys()
    for formatter_name in formatter_order:
        formatter = formatters[formatter_name]
        fs = files_by_formatter.get(formatter_name)
        if fs is None or len(fs) == 0:
            continue
        info(f"{'Formatting' if fix else 'Checking'} {formatter.name} on {len(fs)} files...")
        if not formatter.run(fix=fix, files=fs):
            failed_formatters.append(formatter.name)

    # Print summary
    if failed_formatters:
        error(f"Linting/formatting failed for: {', '.join(failed_formatters)}")
        raise typer.Exit(1)
    else:
        success("All formatting complete")
