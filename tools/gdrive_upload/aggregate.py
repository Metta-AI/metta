"""
File discovery and aggregation for fileset exports.
"""

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Size limits for MVP
MAX_FILE_SIZE = 500 * 1024  # 500 KB per file
MAX_TOTAL_SIZE = 8 * 1024 * 1024  # 8 MB total


def discover_files(includes: List[str], excludes: List[str]) -> List[Path]:
    """
    Discover files matching include patterns, excluding exclude patterns.

    Args:
        includes: List of glob patterns to include
        excludes: List of glob patterns to exclude

    Returns:
        List of Path objects for discovered files
    """
    logger.debug(f"Discovering files with includes: {includes}, excludes: {excludes}")

    all_files = set()

    # Collect all files matching include patterns
    for pattern in includes:
        for path in Path(".").glob(pattern):
            if path.is_file():
                all_files.add(path.resolve())

    # Remove files matching exclude patterns
    excluded_files = set()
    for pattern in excludes:
        for path in Path(".").glob(pattern):
            if path.is_file():
                excluded_files.add(path.resolve())

    final_files = sorted(all_files - excluded_files)
    logger.info(f"Discovered {len(final_files)} files after filtering")

    return final_files


def get_current_commit_sha() -> Optional[str]:
    """Get current git commit SHA if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]  # Short SHA
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def detect_language(file_path: Path) -> str:
    """Detect language for code fence based on file extension."""
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.css': 'css',
        '.html': 'html',
        '.xml': 'xml',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.sh': 'bash',
        '.sql': 'sql',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.m': 'matlab',
        '.tex': 'latex',
        '.dockerfile': 'dockerfile',
    }

    suffix = file_path.suffix.lower()
    return extension_map.get(suffix, 'text')


def read_text_file(file_path: Path) -> Optional[str]:
    """
    Read a text file, returning None if it's binary or too large.

    Args:
        file_path: Path to the file to read

    Returns:
        File content as string, or None if skipped
    """
    try:
        # Check file size
        if file_path.stat().st_size > MAX_FILE_SIZE:
            logger.warning(f"Skipping large file ({file_path.stat().st_size:,} bytes): {file_path}")
            return None

        # Try to read as UTF-8
        content = file_path.read_text(encoding='utf-8')
        return content

    except UnicodeDecodeError:
        logger.warning(f"Skipping binary file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None


def build_markdown(fileset_name: str, files: List[Path]) -> bytes:
    """
    Build aggregated markdown content from discovered files.

    Args:
        fileset_name: Name of the fileset
        files: List of file paths to aggregate

    Returns:
        Aggregated content as bytes
    """
    logger.info(f"Building markdown for {len(files)} files")

    lines = []

    # Header
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    commit_sha = get_current_commit_sha()

    lines.append(f"# Fileset: {fileset_name}")
    lines.append(f"Exported: {timestamp}")
    if commit_sha:
        lines.append(f"Commit: {commit_sha}")
    lines.append("")
    lines.append("> This is an aggregated view of the selected fileset for analysis by Asana's LLMs.")
    lines.append("")

    # Collect file info and content
    file_data = []
    skipped_files = []
    total_size = 0

    for file_path in files:
        content = read_text_file(file_path)
        if content is None:
            skipped_files.append(file_path)
            continue

        file_size = len(content.encode('utf-8'))
        if total_size + file_size > MAX_TOTAL_SIZE:
            logger.warning(f"Total size limit reached, skipping remaining files")
            skipped_files.extend(files[len(file_data):])
            break

        file_data.append({
            'path': file_path,
            'content': content,
            'size': file_size
        })
        total_size += file_size

    logger.info(f"Including {len(file_data)} files ({total_size:,} bytes)")
    if skipped_files:
        logger.info(f"Skipped {len(skipped_files)} files")

    # Table of Contents
    lines.append("## Table of Contents")
    for data in file_data:
        lines.append(f"- {data['path']} ({data['size']:,} bytes)")

    if skipped_files:
        lines.append("")
        lines.append("### Skipped Files")
        for file_path in skipped_files[:10]:  # Limit to first 10
            lines.append(f"- {file_path}")
        if len(skipped_files) > 10:
            lines.append(f"- ... and {len(skipped_files) - 10} more files")

    lines.append("")
    lines.append("---")
    lines.append("")

    # File contents
    for data in file_data:
        file_path = data['path']
        content = data['content']
        language = detect_language(file_path)

        lines.append(f"## {file_path}")
        lines.append(f"```{language}")
        lines.append(content.rstrip())  # Remove trailing whitespace
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Join and normalize line endings
    markdown_content = '\n'.join(lines)
    return markdown_content.encode('utf-8')
