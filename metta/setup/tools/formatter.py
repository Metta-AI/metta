"""Multi-format code formatting tool."""

import subprocess
from typing import Annotated

import typer
from rich.console import Console

from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success, warning

console = Console()
app = typer.Typer(
    help="Format code across multiple file types",
    invoke_without_command=True,
)

# Map file types to their formatter commands
FORMATTERS = {
    "python": {
        "cmd": ["uv", "run", "--active", "ruff", "format", "."],
        "check_cmd": ["uv", "run", "--active", "ruff", "format", "--check", "."],
        "name": "Python (ruff)",
    },
    "json": {
        "cmd": ["bash", "devops/tools/format_json.sh"],
        "name": "JSON",
    },
    "markdown": {
        "cmd": ["bash", "devops/tools/format_md.sh"],
        "name": "Markdown",
    },
    "shell": {
        "cmd": ["bash", "devops/tools/format_sh.sh"],
        "name": "Shell",
    },
    "toml": {
        "cmd": ["bash", "devops/tools/format_toml.sh"],
        "name": "TOML",
    },
    "yaml": {
        "cmd": ["bash", "devops/tools/format_yml.sh"],
        "name": "YAML",
    },
}


@app.callback()
def format_code(
    all: Annotated[bool, typer.Option("--all", "-a", help="Format all file types")] = False,
    file_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Format specific type (python, json, markdown, shell, toml, yaml)",
        ),
    ] = None,
    check: Annotated[bool, typer.Option("--check", help="Check formatting without modifying files")] = False,
):
    """Format code files.

    By default, formats only Python files. Use --all to format all supported file types,
    or --type to format a specific file type.
    """
    repo_root = get_repo_root()

    # Check if C++ formatting is available (mettagrid Makefile exists and has format targets)
    mettagrid_makefile = repo_root / "packages" / "mettagrid" / "Makefile"
    if mettagrid_makefile.exists():
        # Check if Makefile has format targets
        makefile_content = mettagrid_makefile.read_text()
        if "format-fix:" in makefile_content:
            formatter_config = {"cmd": ["make", "-C", "packages/mettagrid", "format-fix"], "name": "C++"}
            if "format-check:" in makefile_content:
                formatter_config["check_cmd"] = ["make", "-C", "packages/mettagrid", "format-check"]
            FORMATTERS["cpp"] = formatter_config

    # Determine which formatters to run
    if all:
        formatters_to_run = list(FORMATTERS.keys())
    elif file_type:
        # Normalize file_type input
        file_type = file_type.lower()
        # Allow "md" as alias for "markdown", "sh" for "shell", "yml" for "yaml"
        aliases = {"md": "markdown", "sh": "shell", "yml": "yaml"}
        file_type = aliases.get(file_type, file_type)

        if file_type not in FORMATTERS:
            error(f"Unknown file type '{file_type}'")
            info(f"Supported types: {', '.join(sorted(FORMATTERS.keys()))}")
            raise typer.Exit(1)
        formatters_to_run = [file_type]
    else:
        # Default: Python only (backward compatible)
        formatters_to_run = ["python"]

    failed = []

    for fmt in formatters_to_run:
        config = FORMATTERS[fmt]
        info(f"Formatting {config['name']}...")

        # Use check_cmd if available and --check is specified, otherwise use regular cmd
        if check and "check_cmd" in config:
            cmd = config["check_cmd"].copy()
        elif check:
            # Formatter doesn't support check mode
            warning(f"--check not supported for {config['name']}, skipping")
            continue
        else:
            cmd = config["cmd"].copy()

        result = subprocess.run(cmd, cwd=repo_root, check=False)

        if result.returncode != 0:
            failed.append(config["name"])

    # Print summary
    if failed:
        error(f"Formatting failed for: {', '.join(failed)}")
        raise typer.Exit(1)
    else:
        success("All formatting complete")


def main():
    """Entry point for the formatter app."""
    app()


if __name__ == "__main__":
    main()
