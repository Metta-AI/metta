"""
File loading and context extraction utilities.

Provides methods for collecting and formatting files from different project structures,
with special handling for READMEs and XML output format.
"""

import fnmatch
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import tiktoken

# Import git helpers
import gitta as git

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


def _build_git_diff_document(base_ref: str, start_path: Path, index: int) -> Optional[Document]:
    """
    Build a Document that contains a Git diff against base_ref.
    Uses git.py helpers. If the repo or base_ref is missing, return a header-only doc.
    """
    repo_root = git.find_root(start_path) or git.find_root(Path.cwd())
    header_title = f"===== Git diff against {base_ref} ====="

    if not repo_root:
        header = [
            header_title,
            "repo: (not a git repository)",
            f"generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "(no diff available)",
        ]
        return Document(index=index, source=f"GIT_DIFF:{base_ref}", content="\n".join(header))

    # Best effort fetch
    git.fetch(repo_root)

    if not git.ref_exists(repo_root, base_ref):
        header = [
            header_title,
            f"repo: {repo_root}",
            f"generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            f"(warning, base ref not found: {base_ref})",
        ]
        return Document(index=index, source=f"GIT_DIFF:{base_ref}", content="\n".join(header))

    diff_text = git.diff(repo_root, base_ref)
    body = diff_text if diff_text.strip() else "(working tree matches base, no changes)"
    lines = [
        header_title,
        f"repo: {repo_root}",
        f"generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        body,
    ]
    return Document(index=index, source=f"GIT_DIFF:{base_ref}@{repo_root}", content="\n".join(lines))


def _find_parent_readmes(path: Path) -> List[Path]:
    """
    Find all README.md files from given path up to git root (or filesystem root).

    Args:
        path: Path to start from

    Returns:
        List of README paths from deepest to shallowest
    """
    readmes = []
    # Start from parent directory for directories, or file's parent for files
    # This ensures we don't include README.md inside the requested directory
    current = path.parent

    # Find the git root to use as our boundary
    git_root = git.find_root(current)

    # Collect READMEs up to and including the git root
    while current != current.parent:  # Not at filesystem root
        # If we have a git root, check if current dir has one and it matches
        if git_root:
            current_git_root = git.find_root(current)
            if not current_git_root or str(current_git_root) != str(git_root):
                # We've gone past the git boundary
                break

        readme_path = current / "README.md"
        if readme_path.exists() and readme_path not in readmes:
            readmes.append(readme_path)

        # If we're at the git root, we're done
        if git_root and str(current) == str(git_root):
            break

        current = current.parent

    # Sort by depth and path
    readmes.sort(key=lambda p: (len(p.parts), str(p)))
    return readmes


def _should_ignore(
    path: Path,
    gitignore_rules: List[str],
    root_dir: Path,
    extensions: Optional[Tuple[str, ...]] = None,
    custom_ignore_dirs: Optional[Tuple[str, ...]] = None,
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

    # Add custom ignore patterns from -i flag
    if custom_ignore_dirs:
        for ignore_dir in custom_ignore_dirs:
            # Add to the ignored_dirs set for consistent pattern matching
            # If it contains a slash, it's a path pattern, otherwise it's a directory name
            if "/" in ignore_dir:
                # For path patterns like "y/z" or "build/cache", check if the relative path matches
                if rel_path == ignore_dir or rel_path.startswith(ignore_dir + os.sep):
                    return True
            else:
                # For simple directory names, add to the set for pattern matching
                ignored_dirs.add(ignore_dir)

    # Check if any part of the path contains ignored directories
    # This applies to both built-in ignored dirs and user-specified directory names
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

    # If filtering by extension for files, and this file doesn't match, ignore it.
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
    custom_ignore_dirs: Optional[Tuple[str, ...]] = None,
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
        if not _should_ignore(path_obj, gitignore_rules, root_dir, extensions, custom_ignore_dirs):
            process_file(path_obj)

    elif path_obj.is_dir():
        for root, dirs, files in os.walk(path_obj):
            # Filter hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Filter directories based on gitignore BEFORE os.walk descends into them
            filtered_dirs = []
            for d in dirs:
                dir_path = Path(os.path.join(root, d))
                if not _should_ignore(dir_path, gitignore_rules, root_dir, extensions, custom_ignore_dirs):
                    filtered_dirs.append(d)
            dirs[:] = filtered_dirs

            # Filter hidden files
            files = [f for f in files if not f.startswith(".")]

            # Filter files based on gitignore and extensions
            files = [
                f
                for f in files
                if not _should_ignore(
                    Path(os.path.join(root, f)), gitignore_rules, root_dir, extensions, custom_ignore_dirs
                )
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
    git_root = git.find_root(current)

    # Search up the directory tree for .gitignore
    while current != current.parent:  # Not at filesystem root
        # If we have a git root, check if current dir is still within it
        if git_root:
            current_git_root = git.find_root(current)
            if not current_git_root or str(current_git_root) != str(git_root):
                # We've gone past the git boundary
                break

        gitignore_path = current / ".gitignore"
        if gitignore_path.exists():
            return gitignore_path

        # If we're at the git root, check it then stop
        if git_root and str(current) == str(git_root):
            break

        current = current.parent

    return None


def _format_document(doc: Document) -> str:
    """Format a document in XML format."""
    lines = [f'<document index="{doc.index}">', f"<source>{doc.source}</source>"]
    if doc.is_readme:
        lines.extend(["<type>readme</type>", "<instructions>", doc.content, "</instructions>"])
    else:
        lines.extend(["<document_content>", doc.content, "</document_content>"])
    lines.append("</document>")
    return "\n".join(lines)


def get_context(
    paths: Optional[List[Union[str, Path]]],
    extensions: Optional[Tuple[str, ...]] = None,
    include_git_diff: bool = False,
    diff_base: str = "origin/main",
    readmes_only: bool = False,
    ignore_dirs: Optional[Tuple[str, ...]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load and format context from specified paths, with basic token counting.

    Args:
        paths: List of paths to load context from, or None for no context
        extensions: Optional tuple of file extensions to filter
        include_git_diff: Whether to include git diff as a virtual file
        diff_base: Base reference for git diff
        readmes_only: Whether to only include README.md files
        ignore_dirs: Optional tuple of directories to ignore

    Returns:
        Tuple of (formatted context string, token info dict)
    """
    from collections import defaultdict

    # Early out only if truly nothing to do and include_git_diff is False
    if not paths and not include_git_diff:
        return "<documents></documents>", {"total_tokens": 0, "total_files": 0}

    # Initialize tokenizer for counting
    encoding = tiktoken.get_encoding("cl100k_base")

    # Staged docs and where they came from
    documents: List[Document] = []
    origin_for_source: Dict[str, Optional[Path]] = {}

    def _stage(doc: Document, origin: Optional[Path]) -> None:
        """
        Add a document and remember which requested path it belongs to.
        origin=None means it is not tied to a requested path
        (for example a Git diff or other virtual document).
        """
        documents.append(doc)
        origin_for_source[doc.source] = origin

    processed_files: Set[Path] = set()
    next_index = 1

    # Process requested paths
    for path_str in paths or []:
        try:
            # Resolve path relative to current directory
            path = resolve_codebase_path(path_str)

            if not path.exists():
                logger.warning(f"Path does not exist: {path}")
                continue

            # Process parent READMEs first
            for readme_path in _find_parent_readmes(path):
                if doc := _load_file(readme_path, next_index, processed_files):
                    _stage(doc, origin=path)  # parent READMEs count toward the requested path
                    processed_files.add(readme_path)
                    next_index += 1

            # Process requested path
            gitignore_path = _find_gitignore(path)
            gitignore_rules = _read_gitignore(str(gitignore_path)) if gitignore_path else []
            gitignore_root = gitignore_path.parent if gitignore_path else None
            new_docs = _collect_files(
                path, gitignore_rules, gitignore_root, processed_files, next_index, extensions, ignore_dirs
            )

            # Stage collected files
            for doc in new_docs:
                if readmes_only and not doc.is_readme:
                    continue
                _stage(doc, origin=path)

            next_index += len([doc for doc in new_docs if not readmes_only or doc.is_readme])

        except Exception as e:
            print(f"Error processing path {path_str}: {e}")

    # Add git diff document if requested
    if include_git_diff:
        start_for_git = resolve_codebase_path(paths[0]) if paths else Path.cwd()
        diff_doc = _build_git_diff_document(diff_base, start_for_git, next_index)
        if diff_doc:
            _stage(diff_doc, origin=None)
            next_index += 1

    # Sort documents (READMEs first, then by path)
    documents.sort(key=lambda d: (not d.is_readme, d.source))

    # Re-index documents
    for i, doc in enumerate(documents, 1):
        doc.index = i

    # One token pass for everything
    total_tokens = 0
    path_summaries: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tokens": 0, "files": 0})
    file_token_counts: Dict[str, int] = {}
    top_level_tokens: Dict[str, int] = {}

    # Work out if there was exactly one requested origin path
    requested_origins = {p for p in origin_for_source.values() if p is not None}
    single_origin: Optional[Path] = next(iter(requested_origins)) if len(requested_origins) == 1 else None

    for doc in documents:
        tokens = len(encoding.encode(doc.content))
        total_tokens += tokens
        file_token_counts[doc.source] = tokens

        origin = origin_for_source.get(doc.source)
        if origin is not None:
            key = str(origin)
            path_summaries[key]["tokens"] += tokens
            path_summaries[key]["files"] += 1

            if single_origin is not None:
                try:
                    relative = Path(doc.source).relative_to(single_origin)
                    if len(relative.parts) > 0:
                        name = relative.parts[0]
                        top_level_tokens[name] = top_level_tokens.get(name, 0) + tokens
                except Exception:
                    pass

    # Generate output
    output_lines = ["<documents>"]
    for doc in documents:
        output_lines.append(_format_document(doc))
    output_lines.append("</documents>")
    content = "\n".join(output_lines)

    # Prepare token info
    token_info = {
        "total_tokens": total_tokens,
        "total_files": len(documents),
        "path_summaries": dict(path_summaries),
        "file_token_counts": file_token_counts,
        "documents": documents,
    }

    # Add top-level summary for single path
    if single_origin is not None and top_level_tokens:
        token_info["top_level_summary"] = top_level_tokens

    return content, token_info
