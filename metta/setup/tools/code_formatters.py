from pathlib import Path

# File type aliases for user convenience (e.g., --type md)
FILE_TYPE_ALIASES = {
    "md": "markdown",
    "sh": "shell",
    "yml": "yaml",
}

# Map file extensions to the formatter types understood by the lint pipeline
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


def detect_file_type(file_path: str) -> str | None:
    """Return the normalized file type for ``file_path`` or ``None`` if unsupported."""

    suffix = Path(file_path).suffix.lower()
    return EXTENSION_TO_TYPE.get(suffix)


def partition_files_by_type(file_paths: list[str]) -> dict[str, list[str]]:
    """Group ``file_paths`` by formatter type, deduplicating entries."""

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
