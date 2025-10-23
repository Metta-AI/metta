"""Multi-format code formatting utilities for metta lint command."""

import subprocess
from pathlib import Path

from metta.setup.utils import info, warning

# File type aliases for user convenience
FILE_TYPE_ALIASES = {
    "md": "markdown",
    "sh": "shell",
    "yml": "yaml",
}

# Map file extensions to formatter types
EXTENSION_TO_TYPE = {
    ".py": "python",
    ".json": "json",
    ".md": "markdown",
    ".sh": "shell",
    ".bash": "shell",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
}


class FormatterConfig:
    """Configuration for a specific file type formatter."""

    def __init__(
        self,
        name: str,
        format_cmd: list[str],
        check_cmd: list[str] | None = None,
    ):
        self.name = name
        self.format_cmd = format_cmd
        self.check_cmd = check_cmd


def get_formatters(repo_root: Path) -> dict[str, FormatterConfig]:
    """Get available formatters for different file types.

    Args:
        repo_root: Repository root directory

    Returns:
        Dictionary mapping file type to FormatterConfig
    """
    formatters = {
        "python": FormatterConfig(
            name="Python (ruff)",
            format_cmd=["uv", "run", "--active", "ruff", "format"],
            check_cmd=["uv", "run", "--active", "ruff", "format", "--check"],
        ),
        "json": FormatterConfig(
            name="JSON",
            format_cmd=["bash", "devops/tools/format_json.sh"],
            check_cmd=None,
        ),
        "markdown": FormatterConfig(
            name="Markdown",
            format_cmd=["bash", "devops/tools/format_md.sh"],
            check_cmd=None,
        ),
        "shell": FormatterConfig(
            name="Shell",
            format_cmd=["bash", "devops/tools/format_sh.sh"],
            check_cmd=None,
        ),
        "toml": FormatterConfig(
            name="TOML",
            format_cmd=["bash", "devops/tools/format_toml.sh"],
            check_cmd=None,
        ),
        "yaml": FormatterConfig(
            name="YAML",
            format_cmd=["bash", "devops/tools/format_yml.sh"],
            check_cmd=None,
        ),
    }

    # Check if C++ formatting is available
    mettagrid_makefile = repo_root / "packages" / "mettagrid" / "Makefile"
    if mettagrid_makefile.exists():
        makefile_content = mettagrid_makefile.read_text()
        if "format-fix:" in makefile_content:
            check_cmd = None
            if "format-check:" in makefile_content:
                check_cmd = ["make", "-C", "packages/mettagrid", "format-check"]
            formatters["cpp"] = FormatterConfig(
                name="C++",
                format_cmd=["make", "-C", "packages/mettagrid", "format-fix"],
                check_cmd=check_cmd,
            )

    return formatters


def detect_file_type(file_path: str) -> str | None:
    """Detect the formatter type for a given file path.

    Args:
        file_path: Path to the file

    Returns:
        File type string (python, json, etc.) or None if unsupported
    """
    suffix = Path(file_path).suffix.lower()
    return EXTENSION_TO_TYPE.get(suffix)


def partition_files_by_type(file_paths: list[str]) -> dict[str, list[str]]:
    """Partition files by their formatter type.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary mapping file type to list of file paths
    """
    files_by_type: dict[str, list[str]] = {}
    seen: set[str] = set()

    for raw_path in file_paths:
        if not raw_path or not (path := raw_path.strip()):
            continue

        if path in seen:
            continue
        seen.add(path)

        file_type = detect_file_type(path)
        if file_type:
            files_by_type.setdefault(file_type, []).append(path)

    return files_by_type


def run_formatter(
    file_type: str,
    formatter: FormatterConfig,
    repo_root: Path,
    *,
    check_only: bool = False,
    files: list[str] | None = None,
) -> bool:
    """Run a formatter for a specific file type.

    Args:
        file_type: Type of files to format (python, json, etc.)
        formatter: Formatter configuration
        repo_root: Repository root directory
        check_only: If True, only check formatting without modifying files
        files: Specific files to format, or None to format all

    Returns:
        True if formatting succeeded, False otherwise
    """
    info(f"Formatting {formatter.name}...")

    # Determine which command to use
    if check_only:
        if formatter.check_cmd is None:
            warning(f"--check not supported for {formatter.name}, skipping")
            return True
        cmd = formatter.check_cmd.copy()
    else:
        cmd = formatter.format_cmd.copy()

    # Add specific files if provided (only for Python/C++)
    if files and file_type in ("python", "cpp"):
        cmd.extend(files)

    result = subprocess.run(cmd, cwd=repo_root, check=False)
    return result.returncode == 0


def normalize_format_type(format_type: str) -> str:
    """Normalize format type string (handle aliases).

    Args:
        format_type: Raw format type string

    Returns:
        Normalized format type
    """
    normalized = format_type.lower()
    return FILE_TYPE_ALIASES.get(normalized, normalized)


def parse_format_types(format_type_str: str, available_formatters: dict[str, FormatterConfig]) -> list[str]:
    """Parse a comma-separated format type string.

    Args:
        format_type_str: Comma-separated format types or "all"
        available_formatters: Available formatter configurations

    Returns:
        List of normalized format type strings

    Raises:
        ValueError: If an unknown format type is specified
    """
    if format_type_str.lower() == "all":
        return list(available_formatters.keys())

    types = []
    for raw_type in format_type_str.split(","):
        normalized = normalize_format_type(raw_type.strip())
        if normalized and normalized not in available_formatters:
            raise ValueError(
                f"Unknown format type '{raw_type}'. Supported types: {', '.join(sorted(available_formatters.keys()))}"
            )
        if normalized:
            types.append(normalized)

    return types
