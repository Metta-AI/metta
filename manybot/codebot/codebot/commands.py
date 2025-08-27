"""
Command implementations for codebot.

This module contains the actual AI-powered commands that can be executed
through the CLI interface.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel
from pydantic_ai import Agent

from .git_utils import find_root
from .models import CommandOutput, ExecutionContext, FileChange
from .summary_models import SummaryResult

logger = logging.getLogger(__name__)


class SummaryCache:
    """Cache summaries to avoid recomputation"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path(".codebot/summaries")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, files: Dict[str, str], token_limit: int) -> str:
        """Generate cache key from file paths + content hashes + token limit"""
        # Create a stable identifier based on files and their content
        file_data = []
        for file_path in sorted(files.keys()):  # Sort for consistent ordering
            content = files[file_path]
            # Use content hash instead of modification time for more reliability
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
            file_data.append(f"{file_path}:{content_hash}")

        # Combine all data into cache key
        cache_input = f"files={';'.join(file_data)}&token_limit={token_limit}"
        cache_key = hashlib.md5(cache_input.encode("utf-8")).hexdigest()[:16]
        return cache_key

    def get_cached_summary(self, files: Dict[str, str], token_limit: int) -> Optional[str]:
        """Get cached summary if it exists"""
        cache_key = self._generate_cache_key(files, token_limit)
        cache_path = self.cache_dir / f"{cache_key}.md"

        if cache_path.exists():
            logger.info(f"Cache hit for key {cache_key}")
            return cache_path.read_text()

        logger.debug(f"Cache miss for key {cache_key}")
        return None

    def store_summary(self, files: Dict[str, str], token_limit: int, summary: str) -> str:
        """Store summary in cache and return cache key"""
        cache_key = self._generate_cache_key(files, token_limit)
        cache_path = self.cache_dir / f"{cache_key}.md"

        cache_path.write_text(summary)
        logger.info(f"Stored summary in cache with key {cache_key}")
        return cache_key

    def get_cache_path(self, files: Dict[str, str], token_limit: int) -> Path:
        """Get the cache file path for given inputs"""
        cache_key = self._generate_cache_key(files, token_limit)
        return self.cache_dir / f"{cache_key}.md"


def _get_codeclip_context():
    """Import and return the get_context function from codeclip."""
    # Add the codeclip directory to the path
    current_dir = os.path.dirname(__file__)
    codeclip_path = os.path.join(current_dir, "..", "codeclip", "codeclip")

    if codeclip_path not in sys.path:
        sys.path.insert(0, codeclip_path)

    try:
        from file import get_context

        return get_context
    except ImportError as e:
        raise ImportError(
            f"Could not import codeclip: {e}. Make sure codeclip is available in the expected location."
        ) from e


class Command(BaseModel):
    """Base class for AI-powered commands using PydanticAI agents."""

    name: str
    prompt_template: str
    default_paths: List[str] = []

    class Config:
        arbitrary_types_allowed = True

    def build_agent(self, result_type=None) -> Agent:
        """Build PydanticAI agent for this command."""
        if result_type is None:
            result_type = CommandOutput

        # Try different constructor patterns for PydanticAI Agent
        try:
            # Try with model as first parameter
            return Agent(model="openai:gpt-4-turbo", result_type=result_type, system_prompt=self.prompt_template)
        except Exception as e:
            logger.debug(f"First constructor failed: {e}")
            try:
                # Try with just model and system_prompt
                return Agent(model="openai:gpt-4-turbo", system_prompt=self.prompt_template)
            except Exception as e2:
                logger.debug(f"Second constructor failed: {e2}")
                # Last fallback - try basic constructor
                return Agent("openai:gpt-4-turbo")

    async def execute(self, context: ExecutionContext) -> CommandOutput:
        """Execute command with the provided context."""
        raise NotImplementedError("Subclasses must implement execute method")

    def _format_context(self, context: ExecutionContext) -> str:
        """Format context for agent consumption."""
        parts = []

        # Format working directory
        parts.append(f"Working Directory: {context.working_directory}")

        # Format files with content
        if context.files:
            parts.append(f"\n=== Files ({len(context.files)} total) ===")
            for file_path, content in context.files.items():
                parts.append(f"\n--- {file_path} ---")
                # Truncate very long files to keep within reasonable limits
                if len(content) > 5000:
                    content = content[:5000] + "\n... (content truncated)"
                parts.append(content)
                parts.append("")  # blank line separator

        # Add git diff if present
        if context.git_diff:
            parts.append("\n=== Git Diff ===")
            parts.append(context.git_diff)

        # Add error output if present
        if context.clipboard:
            parts.append("\n=== Error Output ===")
            parts.append(context.clipboard)

        return "\n".join(parts)


class SummarizeCommand(Command):
    """Implementation of code summarization using PydanticAI agents."""

    name: str = "summarize"
    prompt_template: str = """You are an expert code analyst. \
Analyze the provided code to create a structured summary optimized for AI consumption.

Your task is to:
1. Understand the codebase architecture and purpose
2. Identify key components, classes, and functions
3. Recognize patterns and conventions used
4. Note external dependencies
5. Find entry points and main interfaces
6. Keep the overview concise but informative

Focus on information that would help another AI quickly understand and work with this codebase.
Be accurate and avoid speculation - only include information you can directly observe in the code.

The summary should be comprehensive but focused on the most important aspects of the codebase."""

    default_paths: List[str] = ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.java", "**/*.cpp", "**/*.h"]

    def __init__(self, **data):
        super().__init__(**data)

    async def execute(
        self, context: ExecutionContext, token_limit: int = 2000, no_cache: bool = False
    ) -> CommandOutput:
        """Execute summarization using structured PydanticAI agent."""

        if not context.files:
            return CommandOutput(
                file_changes=[],
                summary="No files provided for summarization",
                success=False,
                metadata={"error": "no_files"},
            )

        # Determine project root and setup cache
        project_root = self._determine_project_root(context)
        cache = SummaryCache(cache_dir=project_root / ".codebot" / "summaries")

        # Check for cached summary first (unless cache is disabled)
        cached_summary = None if no_cache else cache.get_cached_summary(context.files, token_limit)
        if cached_summary:
            cache_key = cache._generate_cache_key(context.files, token_limit)
            cache_path = cache.get_cache_path(context.files, token_limit)

            logger.info(f"Using cached summary (cache key: {cache_key})")

            return CommandOutput(
                file_changes=[
                    FileChange(
                        filepath=str(cache_path),
                        content=cached_summary,
                        metadata={
                            "command": "summarize",
                            "input_files": len(context.files),
                            "token_count": context.token_count,
                            "cached": True,
                            "cache_key": cache_key,
                        },
                    )
                ],
                summary=f"Used cached summary for {len(context.files)} files (cache key: {cache_key[:8]}...)",
                success=True,
                metadata={
                    "output_file": str(cache_path),
                    "cached": True,
                    "cache_key": cache_key,
                    "input_files": len(context.files),
                    "token_limit": token_limit,
                    "input_token_count": context.token_count,
                },
            )

        # Cache miss - create agent with structured result type
        summarizer = self.build_agent(result_type=SummaryResult)

        # Enhance system prompt with token limit information
        enhanced_prompt = f"""{self.prompt_template}

        Keep the total summary under {token_limit} tokens while maintaining completeness.
        Prioritize the most architecturally significant components and patterns.
        """

        # Override the agent's system prompt
        summarizer.system_prompt = enhanced_prompt

        # Prepare context for the agent
        formatted_context = self._format_context(context)

        # Add additional context information
        additional_info = [
            f"Token Limit: {token_limit}",
            f"File Count: {len(context.files)}",
        ]

        summary_context = formatted_context + "\n\n" + "\n".join(additional_info)

        try:
            logger.info(f"Analyzing {len(context.files)} files for summarization...")

            # Run agent with the context
            result = await summarizer.run(summary_context)

            # Handle both structured and unstructured responses
            if hasattr(result, "data") and hasattr(result.data, "to_markdown"):
                # Structured response
                summary: SummaryResult = result.data
                logger.info(
                    f"Generated structured summary with {len(summary.components)} "
                    f"components and {len(summary.patterns)} patterns"
                )
                summary_content = summary.to_markdown()
                stats = summary.to_json_summary()
            else:
                # Unstructured response - treat as raw text
                summary_content = str(result.data) if hasattr(result, "data") else str(result)
                logger.info(f"Generated unstructured summary: {len(summary_content)} characters")

                # Create a minimal mock summary for metadata
                from .summary_models import SummaryResult as SummaryResultType

                # Create a mock SummaryResult for type compatibility
                summary = SummaryResultType(
                    overview="", components=[], external_dependencies=[], patterns=[], entry_points=[]
                )
                stats = summary.to_json_summary()

            # Store in cache and get cache path as output
            cache_key = cache.store_summary(context.files, token_limit, summary_content)
            output_path = cache.get_cache_path(context.files, token_limit)

            # Create file change for the summary
            file_change = FileChange(
                filepath=str(output_path),
                content=summary_content,
                metadata={
                    "command": "summarize",
                    "input_files": len(context.files),
                    "token_count": context.token_count,
                    "summary_stats": stats,
                    "cached": False,
                    "cache_key": cache_key,
                },
            )

            # Generate human-readable summary
            human_summary = self._generate_human_summary(summary, context)

            return CommandOutput(
                file_changes=[file_change],
                summary=human_summary,
                success=True,
                metadata={
                    "output_file": str(output_path),
                    "component_count": len(summary.components),
                    "dependency_count": len(summary.external_dependencies),
                    "pattern_count": len(summary.patterns),
                    "entry_point_count": len(summary.entry_points),
                    "token_limit": token_limit,
                    "input_token_count": context.token_count,
                    "cached": False,
                    "cache_key": cache_key,
                },
            )

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return CommandOutput(
                file_changes=[],
                summary=f"Failed to generate summary: {str(e)}",
                success=False,
                metadata={"error": str(e)},
            )

    def _generate_human_summary(self, summary: SummaryResult, context: ExecutionContext) -> str:
        """Generate a concise human-readable summary of the operation."""

        parts = [
            f"Analyzed {len(context.files)} files",
            f"found {len(summary.components)} key components",
        ]

        if summary.external_dependencies:
            parts.append(f"{len(summary.external_dependencies)} dependencies")

        if summary.patterns:
            parts.append(f"{len(summary.patterns)} patterns")

        if summary.entry_points:
            parts.append(f"{len(summary.entry_points)} entry points")

        base_summary = ", ".join(parts)

        # Add a note about the most important components
        high_priority = [c for c in summary.components if c.importance == "high"]
        if high_priority:
            main_components = ", ".join(c.name for c in high_priority[:3])
            if len(high_priority) > 3:
                main_components += f" (+{len(high_priority) - 3} more)"
            base_summary += f". Main components: {main_components}"

        return base_summary

    def _identify_focus_areas(self, context: ExecutionContext) -> List[str]:
        """Identify areas to focus on during analysis."""
        focus_areas = []

        # Look for common architectural patterns
        file_paths = list(context.files.keys())

        if any("test" in path.lower() for path in file_paths):
            focus_areas.append("testing patterns")

        if any("api" in path.lower() or "server" in path.lower() for path in file_paths):
            focus_areas.append("API and server architecture")

        if any("cli" in path.lower() or "command" in path.lower() for path in file_paths):
            focus_areas.append("CLI interface and commands")

        if any("model" in path.lower() or "schema" in path.lower() for path in file_paths):
            focus_areas.append("data models and schemas")

        if any("util" in path.lower() or "helper" in path.lower() for path in file_paths):
            focus_areas.append("utility functions and helpers")

        return focus_areas

    def _determine_project_root(self, context: ExecutionContext) -> Path:
        """
        Determine the appropriate project root for placing the summary.

        Strategy:
        1. Try to find git root for any of the analyzed file paths
        2. Fall back to the common parent directory of all analyzed paths
        3. Fall back to current working directory as last resort
        """
        if not context.files:
            return Path.cwd()

        # Get all unique directory paths from the analyzed files
        file_paths = [Path(file_path) for file_path in context.files.keys()]

        # Try to find git root for the first file
        first_file_dir = file_paths[0].parent if file_paths[0].is_file() else file_paths[0]
        git_root = find_root(first_file_dir)

        if git_root:
            logger.info(f"Using git root as project root: {git_root}")
            return git_root

        # If no git root, find common parent directory
        if len(file_paths) == 1:
            # Single file/directory - use its parent
            common_parent = file_paths[0].parent if file_paths[0].is_file() else file_paths[0].parent
        else:
            # Multiple paths - find common parent
            try:
                # Get all parent directories
                all_parents = []
                for path in file_paths:
                    path_to_check = path.parent if path.is_file() else path
                    all_parents.extend(path_to_check.parents)
                    all_parents.append(path_to_check)

                # Find the deepest common parent
                all_parents_set = set(all_parents)
                common_parent = max(all_parents_set, key=lambda p: len(p.parts))

                # Verify it's actually common to all paths
                for path in file_paths:
                    path_to_check = path.parent if path.is_file() else path
                    if not str(path_to_check).startswith(str(common_parent)):
                        # Fallback to a more conservative approach
                        common_parent = Path("/")
                        for path in file_paths:
                            path_to_check = path.parent if path.is_file() else path
                            # Find actual common path
                            common_parts = []
                            for a, b in zip(common_parent.parts, path_to_check.parts, strict=False):
                                if a == b:
                                    common_parts.append(a)
                                else:
                                    break
                            if common_parts:
                                common_parent = Path("/".join(common_parts[1:]))  # Skip root /
                            else:
                                common_parent = Path.cwd()
                        break

            except Exception:
                # If anything goes wrong, use current working directory
                common_parent = Path.cwd()

        logger.info(f"Using common parent as project root: {common_parent}")
        return common_parent


async def run_summarize_command(
    paths: Optional[List[Union[str, Path]]] = None,
    token_limit: int = 2000,
    working_dir: Optional[Path] = None,
    no_cache: bool = False,
) -> CommandOutput:
    """
    Convenience function to run the summarize command.

    Args:
        paths: List of file paths to analyze
        token_limit: Maximum tokens for the summary
        working_dir: Working directory for the operation

    Returns:
        CommandOutput with the summarization results
    """
    if working_dir:
        original_cwd = Path.cwd()
        import os

        os.chdir(working_dir)
        try:
            return await _run_summarize_in_dir(paths, token_limit, no_cache)
        finally:
            os.chdir(original_cwd)
    else:
        return await _run_summarize_in_dir(paths, token_limit, no_cache)


async def _run_summarize_in_dir(
    paths: Optional[List[Union[str, Path]]] = None, token_limit: int = 2000, no_cache: bool = False
) -> CommandOutput:
    """Run summarize command in current directory."""

    # Import here to avoid circular imports
    get_context = _get_codeclip_context()

    # Use current directory if no paths provided
    if paths is None:
        paths = ["."]

    # Convert Path objects to strings
    str_paths = [str(p) for p in paths]

    try:
        # Get context using codeclip
        content, token_info = get_context(
            paths=str_paths,
            extensions=(".py", ".ts", ".tsx", ".js", ".java", ".cpp", ".h"),
            include_git_diff=False,
            readmes_only=False,
        )

        # Parse the content to extract individual files from XML format
        import re

        files = {}

        # Find all document blocks
        document_pattern = r"<document[^>]*>.*?</document>"
        documents = re.findall(document_pattern, content, re.DOTALL)

        logger.debug(f"Found {len(documents)} documents in codeclip output")

        for doc in documents:
            # Extract source path
            source_match = re.search(r"<source>(.*?)</source>", doc)
            if not source_match:
                continue

            source_path = source_match.group(1)

            # Find where actual content starts - after the last closing metadata tag
            # Look for patterns like </type>, </instructions>, etc.
            metadata_end_patterns = [r"</type>", r"</instructions>", r"</index>"]
            content_start = -1

            for pattern in metadata_end_patterns:
                match = list(re.finditer(pattern, doc))
                if match:
                    content_start = match[-1].end()
                    break

            if content_start == -1:
                # Fallback: look for any closing tag at the beginning
                first_tag_match = re.search(r"^<[^>]+>", doc)
                if first_tag_match:
                    content_start = first_tag_match.end()
                else:
                    continue

            # Get content after the metadata tags
            file_content = doc[content_start:].strip()

            # Remove the closing </document> tag
            file_content = re.sub(r"</document>\s*$", "", file_content).strip()

            # Skip empty content
            if file_content and len(file_content) > 10:  # Must have some substantial content
                files[source_path] = file_content
                logger.debug(f"Added file {source_path} with {len(file_content)} characters")

        logger.info(f"Successfully parsed {len(files)} files from codeclip output")

        # Create execution context
        context = ExecutionContext(
            files=files,
            working_directory=Path.cwd(),
            token_count=token_info.get("total_tokens", 0),
            metadata=token_info,
        )

        # Create and execute summarize command
        command = SummarizeCommand()
        return await command.execute(context, token_limit=token_limit, no_cache=no_cache)

    except Exception as e:
        logger.error(f"Error in run_summarize_command: {e}")
        return CommandOutput(
            file_changes=[], summary=f"Failed to analyze files: {str(e)}", success=False, metadata={"error": str(e)}
        )
