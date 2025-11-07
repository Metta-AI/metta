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
    format_cmd: list[str]
    check_cmd: list[str]
    extensions: list[str]


def get_formatters() -> dict[str, FormatterConfig]:
    formatters = {
        "python": FormatterConfig(
            name="Python (ruff)",
            format_cmd=["uv", "run", "ruff", "format"],
            check_cmd=["uv", "run", "--active", "ruff", "format", "--check"],
            extensions=[".py"],
        ),
        "json": FormatterConfig(
            name="JSON",
            format_cmd=["bash", "devops/tools/format_json.sh"],
            check_cmd=["bash", "devops/tools/format_json.sh", "--check"],
            extensions=[".json"],
        ),
        "markdown": FormatterConfig(
            name="Markdown",
            format_cmd=["bash", "devops/tools/format_md.sh"],
            check_cmd=["bash", "devops/tools/format_md.sh", "--check"],
            extensions=[".md"],
        ),
        "shell": FormatterConfig(
            name="Shell",
            format_cmd=["bash", "devops/tools/format_sh.sh"],
            check_cmd=["bash", "devops/tools/format_sh.sh", "--check"],
            extensions=[".sh", ".bash"],
        ),
        "toml": FormatterConfig(
            name="TOML",
            format_cmd=["bash", "devops/tools/format_toml.sh"],
            check_cmd=["bash", "devops/tools/format_toml.sh", "--check"],
            extensions=[".toml"],
        ),
        "yaml": FormatterConfig(
            name="YAML",
            format_cmd=["bash", "devops/tools/format_yml.sh"],
            check_cmd=["bash", "devops/tools/format_yml.sh", "--check"],
            extensions=[".yaml", ".yml"],
        ),
        "cpp": FormatterConfig(
            name="C++",
            format_cmd=["make", "-C", "packages/mettagrid", "format-fix"],
            check_cmd=["make", "-C", "packages/mettagrid", "format-check"],
            extensions=[".cpp", ".hpp", ".h"],
        ),
    }

    return formatters


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
        metta lint                    # Format and lint all detected files
        metta lint --fix              # Format and lint with auto-fix
        metta lint --staged --fix     # Format and lint only staged files
        metta lint --files path/to/file1 --files path/to/file2 --fix
    """
    # Get available formatters
    formatters = get_formatters()

    # Determine which files to process
    target_files: list[str] | None = None
    files_by_type: DefaultDict[str, set[str]] | None = None
    if files is not None:
        target_files = files
    elif staged:
        target_files = [fname for status, fname in git.get_uncommitted_files_by_status() if status[1] == "M"]

    if target_files is not None:
        files_by_type = defaultdict(set)
        for f in target_files:
            ext = Path(f).suffix.lower()
            for formatter in formatters.values():
                if ext in formatter.extensions:
                    files_by_type[formatter.name].add(f)

    failed_formatters = []

    # Run formatters for each type
    for file_type in (files_by_type.keys() if files_by_type else formatters.keys()) & formatters.keys():
        formatter = formatters[file_type]
        info(f"{'Formatting' if fix else 'Checking'} {formatter.name}...")
        cmd = formatter.format_cmd.copy() if not fix else formatter.check_cmd.copy()
        if files_by_type is not None and (applicable_files := files_by_type.get(file_type)):
            cmd.extend(applicable_files)
        result = subprocess.run(cmd, cwd=get_repo_root(), check=False)
        if result.returncode != 0:
            failed_formatters.append(formatter.name)

    # Print summary
    if failed_formatters:
        error(f"Linting/formatting failed for: {', '.join(failed_formatters)}")
        raise typer.Exit(1)
    else:
        success("All formatting complete")
