"""
File loading and context extraction utilities.

Provides methods for collecting and formatting files from different project structures,
with special handling for READMEs and support for both XML and raw output formats.
"""

import fnmatch
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a file's content and metadata for context output."""

    index: int
    source: str
    content: str
    is_readme: bool = False


def resolve_codebase_path(path_str: Union[str, Path]) -> Path:
    """
    Convert path string to an absolute path relative to current working directory.

    Args:
        path_str: Path to resolve

    Returns:
        Resolved absolute path
    """
    path_obj = Path(str(path_str))

    # If path is absolute, just return it
    if path_obj.is_absolute():
        return path_obj.resolve()

    # All relative paths are resolved against current working directory
    return (Path.cwd() / path_obj).resolve()


def _find_git_root(start_path: Path) -> Optional[Path]:
    """
    Find the root of the git repository containing the given path.

    Args:
        start_path: Path to start searching from

    Returns:
        Path to git root directory, or None if not in a git repo
    """
    current = start_path if start_path.is_dir() else start_path.parent

    # Search up the directory tree for .git directory
    while current != current.parent:
        if (current / ".git").is_dir():
            return current
        current = current.parent

    return None


def _find_parent_readmes(path: Path) -> List[Path]:
    """
    Find all README.md files from given path up to git root (or filesystem root).

    Args:
        path: Path to start from

    Returns:
        List of README paths from deepest to shallowest
    """
    readmes = []
    current = path if path.is_dir() else path.parent

    # Find the git root to use as our boundary
    git_root = _find_git_root(current)
    stop_at = git_root if git_root else Path("/")

    # Collect READMEs up to the boundary
    while True:
        readme_path = current / "README.md"
        if readme_path.exists() and readme_path not in readmes:
            readmes.append(readme_path)

        # Stop if we've reached our boundary or can't go higher
        if current == stop_at or current == current.parent:
            break

        current = current.parent

    # Sort by depth and path
    readmes.sort(key=lambda p: (len(p.parts), str(p)))
    return readmes


def _should_ignore(
    path: Path, gitignore_rules: List[str], root_dir: Path, extensions: Optional[Tuple[str, ...]] = None
) -> bool:
    rel_path = os.path.relpath(str(path), root_dir)
    basename = path.name

    # Common dependency/build directories to always ignore
    ignored_dirs = {
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "virtualenv",
        ".tox",
        "dist",
        "build",
        "target",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        ".coverage",
        "site-packages",
        "vendor",
        "vendors",
        "bower_components",
        ".bundle",
        "deps",
        "dependencies",
        ".cargo",
        ".gradle",
        ".idea",
        ".vscode",
        ".vs",
        "obj",
        "bin",
        "out",
        ".next",
        ".nuxt",
        ".cache",
        ".parcel-cache",
        ".turbo",
        ".vercel",
        ".netlify",
        ".serverless",
        "tmp",
        "temp",
        "logs",
        ".git",
        ".svn",
        ".hg",
    }

    # Check if any part of the path contains ignored directories
    path_parts = Path(rel_path).parts
    for part in path_parts:
        if part in ignored_dirs:
            return True

    # Always skip binary or data files
    if _should_ignore_file_type(path):
        return True

    # Always include README.md
    if basename == "README.md":
        return False

    # If filtering by extension for files, and this file doesn’t match, ignore it.
    if extensions and path.is_file() and not any(path.name.endswith(ext) for ext in extensions):
        return True

    for rule in gitignore_rules:
        rule = rule.strip()
        if not rule or rule.startswith("#"):
            continue

        # If the rule contains a slash, treat it as relative to the project_dir.
        if "/" in rule:
            # For rules starting with a slash, strip it and check if rel_path starts with the pattern.
            if rule.startswith("/"):
                pattern = rule.lstrip("/")
                # If rule ends with a slash, match directory prefixes.
                if rule.endswith("/"):
                    if rel_path.startswith(pattern):
                        return True
                else:
                    # Use fnmatch to allow wildcards.
                    if fnmatch.fnmatch(rel_path, pattern):
                        return True
            else:
                # Rule with a slash but not anchored; apply fnmatch against the entire relative path.
                # Special handling for directory rules ending with /
                if rule.endswith("/"):
                    dir_pattern = rule.rstrip("/")
                    # Match the directory itself or any path within it
                    if rel_path == dir_pattern or rel_path.startswith(dir_pattern + "/"):
                        return True
                elif fnmatch.fnmatch(rel_path, rule):
                    return True
        else:
            # For rules without a slash:
            # If the rule appears as a glob pattern, match against basename and anywhere in rel_path.
            if any(ch in rule for ch in "*?[]"):
                if fnmatch.fnmatch(basename, rule) or fnmatch.fnmatch(rel_path, f"*{rule}*"):
                    return True
            else:
                # Plain string: if the rule appears anywhere in basename or rel_path, ignore.
                if rule in basename or rule in rel_path:
                    return True
    return False


def _should_ignore_file_type(path: Path) -> bool:
    """Check if a file should be ignored based on its type (extension, name, or pattern)."""
    # Binary and compiled extensions
    binary_extensions = {
        # Python compiled
        ".pyc",
        ".pyo",
        ".pyd",
        # Native libraries
        ".so",
        ".dll",
        ".dylib",
        ".node",
        # Executables
        ".exe",
        ".bin",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".z",
        # Images
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".svg",
        ".ico",
        ".webp",
        ".tiff",
        # Fonts
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        # ML model files
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".pkl",
        ".pickle",
        ".joblib",
        ".npy",
        # Database files
        ".db",
        ".sqlite",
        ".sqlite3",
        ".duckdb",
        # Data formats
        ".parquet",
        ".feather",
        ".arrow",
        ".hdf5",
        ".h5",
        ".pbf",
        # Package files
        ".whl",
        ".egg",
        # Misc binary
        ".wandb",
    }

    # Non-source text extensions
    data_extensions = {
        ".log",
        ".cache",
        ".tmp",
        ".temp",
        ".bak",
        ".backup",
        ".swp",
        ".swo",
        ".swn",
        ".wal",
    }

    # Specific filenames to ignore
    ignored_filenames = {
        # Lock files
        "uv.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
        # Cache files
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        # Build artifacts
        ".coverage",
        "coverage.xml",
    }

    # Check extensions
    ext = path.suffix.lower()
    if ext in binary_extensions or ext in data_extensions:
        return True

    # Check exact filenames
    if path.name in ignored_filenames:
        return True

    # Check patterns
    ignored_patterns = [
        "*_cache.json",
        "*.cache",
        "*.egg-info",
        "*~",  # Editor temp files
        ".#*",  # Emacs lock files
    ]

    for pattern in ignored_patterns:
        if fnmatch.fnmatch(path.name, pattern):
            return True

    return False


def _load_file(
    path: Path,
    index: int,
    processed_files: Set[Path],
) -> Optional[Document]:
    """Load a single file into a Document if it meets criteria."""
    if path in processed_files:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return Document(index=index, source=str(path), content=f.read(), is_readme=path.name.endswith("README.md"))
    except UnicodeDecodeError:
        logger.warning(f"Skipping file {path} due to UnicodeDecodeError")
        return None
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None


def _matches_extensions(path: Path, extensions: Optional[Tuple[str, ...]] = None) -> bool:
    """Check if a file path matches any of the given extensions."""
    if extensions is None or len(extensions) == 0:
        return True
    logger.info(f"Checking if {path} matches extensions {extensions}")
    logger.debug(f"Any extensions match? {any(path.name.endswith(ext) for ext in extensions)}")
    return path.name.endswith("README.md") or any(path.name.endswith(ext) for ext in extensions)


def _collect_files(
    path: Path,
    gitignore_rules: List[str],
    gitignore_root: Optional[Path],
    processed_files: Set[Path],
    next_index: int,
    extensions: Optional[Tuple[str, ...]] = None,
) -> List[Document]:
    """Recursively collect files into Document objects."""
    documents = []
    current_index = next_index
    path_obj = Path(path)
    # Use the gitignore root if provided, otherwise use the path's parent
    root_dir = gitignore_root if gitignore_root else (path_obj if path_obj.is_dir() else path_obj.parent)

    def process_file(file_path: Path) -> None:
        nonlocal current_index
        if doc := _load_file(file_path, current_index, processed_files):
            documents.append(doc)
            processed_files.add(file_path)
            current_index += 1

    if path_obj.is_file():
        if not _should_ignore(path_obj, gitignore_rules, root_dir, extensions):
            process_file(path_obj)

    elif path_obj.is_dir():
        for root, dirs, files in os.walk(path_obj):
            # Filter hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Filter directories based on gitignore BEFORE os.walk descends into them
            filtered_dirs = []
            for d in dirs:
                dir_path = Path(os.path.join(root, d))
                if not _should_ignore(dir_path, gitignore_rules, root_dir, extensions):
                    filtered_dirs.append(d)
            dirs[:] = filtered_dirs

            # Filter hidden files
            files = [f for f in files if not f.startswith(".")]

            # Filter files based on gitignore and extensions
            files = [
                f
                for f in files
                if not _should_ignore(Path(os.path.join(root, f)), gitignore_rules, root_dir, extensions)
            ]

            for file_name in sorted(files):
                process_file(Path(os.path.join(root, file_name)))

    return documents


def _read_gitignore(path: str) -> List[str]:
    """Return lines from .gitignore for ignoring certain files/directories."""
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                return [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except Exception:
            return []
    return []


def _find_gitignore(start_path: Path) -> Optional[Path]:
    """
    Find the nearest .gitignore file by searching up the directory tree.

    Args:
        start_path: Path to start searching from

    Returns:
        Path to .gitignore file, or None if not found
    """
    current = start_path if start_path.is_dir() else start_path.parent

    # Find the git root to use as our boundary
    git_root = _find_git_root(current)
    stop_at = git_root if git_root else Path("/")

    # Search up the directory tree for .gitignore
    while True:
        gitignore_path = current / ".gitignore"
        if gitignore_path.exists():
            return gitignore_path

        # Stop if we've reached our boundary or can't go higher
        if current == stop_at or current == current.parent:
            break

        current = current.parent

    return None


def _format_document(doc: Document, raw: bool) -> str:
    """Format a document according to output format."""
    if raw:
        lines = [doc.source, "---"]
        if doc.is_readme:
            lines.append("### README START ###")
        lines.append(doc.content)
        if doc.is_readme:
            lines.append("### README END ###")
        lines.append("---")
    else:
        lines = [f'<document index="{doc.index}">', f"<source>{doc.source}</source>"]
        if doc.is_readme:
            lines.extend(["<type>readme</type>", "<instructions>", doc.content, "</instructions>"])
        else:
            lines.extend(["<document_content>", doc.content, "</document_content>"])
        lines.append("</document>")

    return "\n".join(lines)


def get_context(
    paths: Optional[List[Union[str, Path]]], raw: bool = False, extensions: Optional[Tuple[str, ...]] = None
) -> str:
    """
    Load and format context from specified paths.

    Args:
        paths: List of paths to load context from, or None for no context
        raw: Whether to use raw format instead of XML
        extensions: Optional tuple of file extensions to filter

    Returns:
        Formatted context string
    """

    # Handle empty paths
    if not paths:
        return "" if raw else "<documents></documents>"

    processed_files: Set[Path] = set()
    documents = []
    next_index = 1

    for path_str in paths:
        try:
            # Resolve path relative to current directory
            path = resolve_codebase_path(path_str)

            if not path.exists():
                logger.warning(f"Path does not exist: {path}")
                continue

            # Process parent READMEs first
            for readme_path in _find_parent_readmes(path):
                if doc := _load_file(readme_path, next_index, processed_files):
                    documents.append(doc)
                    processed_files.add(readme_path)
                    next_index += 1

            # Process requested path
            gitignore_path = _find_gitignore(path)
            gitignore_rules = _read_gitignore(str(gitignore_path)) if gitignore_path else []
            gitignore_root = gitignore_path.parent if gitignore_path else None
            new_docs = _collect_files(path, gitignore_rules, gitignore_root, processed_files, next_index, extensions)
            next_index += len(new_docs)
            documents.extend(new_docs)

        except Exception as e:
            print(f"Error processing path {path_str}: {e}")

    # Sort documents (READMEs first, then by path)
    documents.sort(key=lambda d: (not d.is_readme, d.source))

    # Re-index documents
    for i, doc in enumerate(documents, 1):
        doc.index = i

    # Generate output
    output_lines = []
    if not raw:
        output_lines.append("<documents>")

    for doc in documents:
        output_lines.append(_format_document(doc, raw))

    if not raw:
        output_lines.append("</documents>")

    return "\n".join(output_lines)
