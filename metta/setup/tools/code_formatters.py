import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Annotated, DefaultDict, Optional

import typer
from pydantic import BaseModel

import gitta as git
from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success


class FormatterConfig(BaseModel):
    name: str
    format_cmds: tuple[tuple[str, ...], ...]
    check_cmds: tuple[tuple[str, ...], ...]
    extensions: tuple[str, ...] = ()

    def run(self, fix: bool = False, files: set[str] | None = None) -> bool:
        commands = self.check_cmds if not fix else self.format_cmds
        if not commands:
            return False
        file_args = sorted(files) if files else []
        for base_cmd in commands:
            cmd = list(base_cmd)
            if file_args:
                cmd.extend(file_args)
            info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=get_repo_root(), check=False)
            if result.returncode != 0:
                return False
        return True


def get_formatters() -> dict[str, FormatterConfig]:
    return {
        f.name: f
        for f in [
            FormatterConfig(
                name="Python (ruff)",
                format_cmds=(
                    ("uv", "run", "ruff", "check", "--fix"),
                    ("uv", "run", "ruff", "format"),
                ),
                check_cmds=(
                    ("uv", "run", "ruff", "check"),
                    ("uv", "run", "ruff", "format", "--check"),
                ),
                extensions=(".py",),
            ),
            FormatterConfig(
                name="JSON",
                format_cmds=(("bash", "devops/tools/format_json.sh"),),
                check_cmds=(("bash", "devops/tools/format_json.sh", "--check"),),
                extensions=(".json", ".jsonc", ".code-workspace"),
            ),
            FormatterConfig(
                name="Markdown",
                format_cmds=(("bash", "devops/tools/format_md.sh"),),
                check_cmds=(("bash", "devops/tools/format_md.sh", "--check"),),
                extensions=(".md",),
            ),
            FormatterConfig(
                name="Shell",
                format_cmds=(("bash", "devops/tools/format_sh.sh"),),
                check_cmds=(("bash", "devops/tools/format_sh.sh", "--check"),),
                extensions=(".sh", ".bash"),
            ),
            FormatterConfig(
                name="TOML",
                format_cmds=(("bash", "devops/tools/format_toml.sh"),),
                check_cmds=(("bash", "devops/tools/format_toml.sh", "--check"),),
                extensions=(".toml",),
            ),
            FormatterConfig(
                name="YAML",
                format_cmds=(("bash", "devops/tools/format_yml.sh"),),
                check_cmds=(("bash", "devops/tools/format_yml.sh", "--check"),),
                extensions=(".yaml", ".yml"),
            ),
            FormatterConfig(
                name="C++",
                format_cmds=(("make", "-C", "packages/mettagrid", "format-fix"),),
                check_cmds=(("make", "-C", "packages/mettagrid", "format-check"),),
                extensions=(".cpp", ".hpp", ".h"),
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
    target_files: list[str] | None = None
    files_by_formatter: DefaultDict[str, set[str]] | None = None
    if files is not None:
        target_files = files
    elif staged:
        target_files = [fname for status, fname in git.get_uncommitted_files_by_status() if status[0] in ("M", "A", "R")]

    if target_files is not None:
        files_by_formatter = defaultdict(set)
        for f in target_files:
            ext = Path(f).suffix.lower()
            for formatter_name in formatters:
                if ext in formatters[formatter_name].extensions:
                    files_by_formatter[formatter_name].add(f)
                    break

    failed_formatters = []

    # Run formatters for each type
    for formatter_name in files_by_formatter.keys() if files_by_formatter is not None else formatters.keys():
        formatter = formatters[formatter_name]
        fs = files_by_formatter.get(formatter_name) if files_by_formatter is not None else None
        info(f"{'Formatting' if fix else 'Checking'} {formatter.name} on {len(fs) if fs else 'all'} files...")
        if not formatter.run(fix=fix, files=fs):
            failed_formatters.append(formatter.name)

    # Print summary
    if failed_formatters:
        error(f"Linting/formatting failed for: {', '.join(failed_formatters)}")
        raise typer.Exit(1)
    else:
        success("All formatting complete")
