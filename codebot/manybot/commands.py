"""
Command implementations for codebot.

This module contains the actual AI-powered commands that can be executed
through the CLI interface.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel
from pydantic_ai import Agent

from .command_models import (
    DebugResult,
    FixResult,
    ImplementationResult,
    RefactoringResult,
    TestGenerationResult,
)
from .git_layer import GitLayer, GitLayerError
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
    codeclip_path = os.path.join(current_dir, "..", "codeclip")

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

    async def execute(
        self, context: ExecutionContext, mode: Optional[str] = None, auto_commit: bool = True
    ) -> CommandOutput:
        """Execute command using appropriate mode."""

        # Execute based on mode
        if mode == "interactive":
            return await self._execute_interactive(context)
        elif mode == "claudesdk":
            return await self._execute_claudesdk(context)
        else:
            # Default oneshot execution with PydanticAI
            result = await self._execute_oneshot(context)

        # Auto-commit changes if requested and we have a git layer
        if auto_commit and result.success and result.file_changes:
            try:
                git_layer = GitLayer(context.working_directory)

                # Apply changes first
                for change in result.file_changes:
                    change.apply()

                # Commit with AI tracking
                git_layer.commit_ai_changes(command=self.name, changes=result.file_changes, description=result.summary)

                # Update metadata to indicate commit was made
                result.metadata["git_commit"] = True

            except (GitLayerError, Exception) as e:
                logger.warning(f"Failed to auto-commit changes: {e}")
                result.metadata["git_commit"] = False

        return result

    async def _execute_oneshot(self, context: ExecutionContext) -> CommandOutput:
        """Default oneshot execution with PydanticAI."""
        raise NotImplementedError("Subclasses must implement _execute_oneshot method")

    async def _execute_claudesdk(self, context: ExecutionContext) -> CommandOutput:
        """Execute using autonomous Claude SDK with file editing permissions."""
        try:
            # Get file paths from context
            file_paths = list(context.files.keys())
            if not file_paths:
                return CommandOutput(
                    success=False,
                    summary="No files provided for Claude SDK execution",
                    metadata={"error": "empty_context"},
                )

            # Build simple autonomous prompt (no file content needed)
            prompt = self._build_autonomous_prompt(file_paths, context)

            # Determine allowed directories (parent dirs of all files)
            allowed_dirs = set()
            for file_path in file_paths:
                parent_dir = str(Path(file_path).parent)
                allowed_dirs.add(parent_dir)

            # Build autonomous Claude command
            cmd = ["claude", "-p"]

            # Add directory permissions
            for dir_path in allowed_dirs:
                cmd.extend(["--add-dir", dir_path])

            # Add other options
            cmd.extend(["--allowed-tools", "Edit", "--permission-mode", "acceptEdits", "--output-format", "json"])

            # Prompt must be the last argument
            cmd.append(prompt)

            # Debug logging
            logger.info(f"Executing autonomous Claude SDK command: {cmd[:8]}... (truncated)")
            logger.info(f"Allowed directories: {list(allowed_dirs)}")
            logger.info(f"Target files: {file_paths}")
            logger.info(f"Prompt length: {len(prompt)} characters")

            # Take snapshot of current git state
            git_before = self._get_git_status()

            # Execute claude -p autonomously
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Debug logging
            logger.info(f"Claude SDK return code: {result.returncode}")
            logger.info(f"Claude SDK stderr: {result.stderr}")

            if result.returncode != 0:
                return CommandOutput(
                    success=False,
                    summary=f"Autonomous Claude SDK execution failed: {result.stderr}",
                    metadata={"error": result.stderr},
                )

            # Parse Claude's response for summary/reasoning
            try:
                claude_response = json.loads(result.stdout)
                summary = claude_response.get("result", "File changes completed")
                logger.info(f"Claude response: {summary[:200]}...")
            except json.JSONDecodeError:
                summary = "File changes completed autonomously"

            # Detect changes using git
            git_after = self._get_git_status()
            changed_files = self._detect_file_changes(git_before, git_after)

            if not changed_files:
                return CommandOutput(
                    success=True,
                    summary="No changes detected after Claude execution",
                    metadata={"claude_response": summary},
                )

            # Create FileChange objects from git changes
            file_changes = []
            for file_path in changed_files:
                try:
                    if Path(file_path).exists():
                        # File was modified or created
                        content = Path(file_path).read_text(encoding="utf-8")
                        file_changes.append(FileChange(filepath=file_path, content=content, operation="write"))
                    else:
                        # File was deleted
                        file_changes.append(FileChange(filepath=file_path, content="", operation="delete"))
                except Exception as e:
                    logger.warning(f"Could not read changed file {file_path}: {e}")

            logger.info(f"Detected {len(file_changes)} file changes: {[fc.filepath for fc in file_changes]}")

            return CommandOutput(
                success=True,
                file_changes=file_changes,
                summary=f"Autonomously modified {len(file_changes)} file{'s' if len(file_changes) != 1 else ''}",
                metadata={
                    "claude_response": summary,
                    "changed_files": [fc.filepath for fc in file_changes],
                    "execution_mode": "autonomous",
                },
            )

        except subprocess.TimeoutExpired:
            return CommandOutput(
                success=False, summary="Autonomous Claude SDK execution timed out", metadata={"timeout": 300}
            )
        except FileNotFoundError:
            return CommandOutput(
                success=False,
                summary="Claude SDK (claude -p) not found. Please install the Claude SDK.",
                metadata={"missing_dependency": "claude"},
            )

    async def _execute_interactive(self, context: ExecutionContext) -> CommandOutput:
        """Execute in interactive mode (launches Claude Code)."""
        # This would launch Claude Code with the context
        # For now, return a placeholder
        return CommandOutput(
            success=False, summary="Interactive mode not yet implemented", metadata={"mode": "interactive"}
        )

    def _build_autonomous_prompt(self, file_paths: List[str], context: ExecutionContext) -> str:
        """Build a simple autonomous prompt with file paths (no content needed)."""
        files_str = ", ".join(file_paths)

        # Include clipboard context if available (for debug-tests)
        clipboard_context = ""
        if hasattr(context, "clipboard") and context.clipboard:
            clipboard_context = f"\n\nError output from clipboard:\n{context.clipboard}"

        prompt = f"""{self.prompt_template}

Target files: {files_str}

Claude, please read and {self.name} the specified files following the requirements above.{clipboard_context}

Important: You have direct file access - read the files yourself, make the improvements, and save your changes."""

        return prompt

    def _get_git_status(self) -> Dict[str, str]:
        """Get current git status as a dict of filename -> status."""
        try:
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)

            status_dict = {}
            for line in result.stdout.strip().split("\n"):
                if line:
                    # Git porcelain format: status characters followed by space then filename
                    # Handle different formats: "XY filename", "X filename", etc.
                    if len(line) >= 3 and line[2] == " ":
                        # Standard format: "XY filename"
                        status = line[:2]
                        filename = line[3:]
                    elif len(line) >= 2 and line[1] == " ":
                        # Single char status: "X filename"
                        status = line[0] + " "  # Normalize to 2 chars
                        filename = line[2:]
                    else:
                        # Fallback - split on first space
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            status = parts[0].ljust(2)  # Normalize to 2 chars
                            filename = parts[1]
                        else:
                            continue  # Skip malformed lines

                    # Debug: logger.info(f"Git parsed: {filename} with status {status}")
                    status_dict[filename] = status

            return status_dict
        except subprocess.CalledProcessError:
            return {}

    def _detect_file_changes(self, before: Dict[str, str], after: Dict[str, str]) -> List[str]:
        """Detect which files changed between two git status snapshots."""
        # Debug: logger.info(f"Detecting changes - before: {before}, after: {after}")
        changed_files = []

        # Files that are new or have different status
        for filename, status in after.items():
            if filename not in before or before[filename] != status:
                # Debug: logger.info(f"Change detected - filename: {repr(filename)}, status: {repr(status)}")
                changed_files.append(filename)

        # Files that were removed
        for filename in before:
            if filename not in after:
                # Debug: logger.info(f"Removal detected - filename: {repr(filename)}")
                changed_files.append(filename)

        # Debug: logger.info(f"Final changed_files: {changed_files}")
        return changed_files

    def _build_claudesdk_prompt(self, context: ExecutionContext) -> str:
        """Build prompt for Claude SDK execution."""
        formatted_context = self._format_context(context)

        prompt = f"""{self.prompt_template}

Context:
{formatted_context}

IMPORTANT: Return ONLY a valid JSON object. Do not include any explanations, markdown formatting, or ```json delimiters.

Your response must be a single JSON object with these exact fields:
- file_changes: array of changes to make, each with filepath, content, and operation
- summary: brief description of changes made
- reasoning: explanation of why these changes were made

Requirements:
- All file paths must be relative to the working directory
- Content must include complete, working code for each file
- Use "write" operation for creating/modifying files, "delete" for removing files
- Return ONLY the JSON object, nothing else"""

        return prompt

    def _extract_file_changes_from_claudesdk_output(self, output: dict) -> CommandOutput:
        """Extract file changes from Claude SDK JSON output."""
        try:
            file_changes = []

            for change_data in output.get("file_changes", []):
                file_change = FileChange(
                    filepath=change_data["filepath"],
                    content=change_data["content"],
                    operation=change_data.get("operation", "write"),
                )
                file_changes.append(file_change)

            return CommandOutput(
                file_changes=file_changes,
                summary=output.get("summary", "Changes made via Claude SDK"),
                metadata={"mode": "claudesdk", "reasoning": output.get("reasoning", "")},
            )

        except KeyError as e:
            return CommandOutput(
                success=False, summary=f"Invalid Claude SDK output format: missing {e}", metadata={"raw_output": output}
            )

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


# New Command Implementations


class TestCommand(Command):
    """Generate comprehensive tests using PydanticAI."""

    name: str = "test"
    prompt_template: str = """You are an expert test engineer. Generate comprehensive tests for the provided code.

Your task is to:
1. Analyze the code structure and identify testable components
2. Create unit tests for all public functions and methods
3. Include edge cases, error scenarios, and boundary conditions
4. Follow testing best practices and conventions
5. Generate clean, readable test code

Follow these guidelines:
- Write clear test names that describe what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use appropriate test fixtures and setup/teardown
- Follow the project's existing test patterns if any
- Ensure tests are independent and can run in any order

Generate tests that provide good coverage while being maintainable and reliable."""

    default_paths: List[str] = ["**/*.py", "!test_*.py", "!*_test.py", "!tests/**"]

    async def _execute_oneshot(self, context: ExecutionContext) -> CommandOutput:
        """Execute test generation using structured PydanticAI agent."""

        if not context.files:
            return CommandOutput(
                success=False, summary="No files provided for test generation", metadata={"error": "no_files"}
            )

        # Create agent with structured result type
        test_agent = self.build_agent(result_type=TestGenerationResult)

        # Prepare context
        formatted_context = self._format_context(context)

        # Add test-specific instructions
        test_context = f"""{formatted_context}

Additional Instructions:
- Focus on the most important and complex functions first
- Create tests that verify both functionality and edge cases
- Use descriptive test names that explain what is being verified
- Include error handling tests where appropriate
- Generate realistic test data and scenarios"""

        try:
            logger.info(f"Generating tests for {len(context.files)} files...")

            # Run agent
            result = await test_agent.run(test_context)

            if hasattr(result, "data"):
                test_result: TestGenerationResult = result.data

                # Generate test files
                file_changes = []
                for source_file in context.files.keys():
                    # Determine test file name
                    test_file_path = self._get_test_file_path(source_file)

                    # Generate test content
                    test_content = test_result.to_test_file(target_module=self._get_module_name(source_file))

                    file_changes.append(
                        FileChange(
                            filepath=test_file_path,
                            content=test_content,
                            metadata={
                                "command": "test",
                                "source_file": source_file,
                                "test_count": len(test_result.test_cases),
                            },
                        )
                    )

                return CommandOutput(
                    file_changes=file_changes,
                    summary=f"Generated {len(test_result.test_cases)} tests for {len(context.files)} files",
                    metadata={
                        "test_cases": len(test_result.test_cases),
                        "files_tested": len(context.files),
                        "test_framework": test_result.test_framework,
                    },
                )
            else:
                return CommandOutput(
                    success=False,
                    summary="Failed to generate structured test result",
                    metadata={"error": "invalid_response"},
                )

        except Exception as e:
            logger.error(f"Error in test generation: {e}")
            return CommandOutput(
                success=False, summary=f"Failed to generate tests: {str(e)}", metadata={"error": str(e)}
            )

    def _get_test_file_path(self, source_file: str) -> str:
        """Generate test file path from source file."""
        path = Path(source_file)
        if path.suffix == ".py":
            return f"test_{path.stem}.py"
        else:
            return f"test_{path.name}.py"

    def _get_module_name(self, source_file: str) -> str:
        """Generate module name from source file path."""
        path = Path(source_file)
        return path.stem


class RefactorCommand(Command):
    """Refactor code to improve quality without changing behavior."""

    name: str = "refactor"
    prompt_template: str = """You are an expert code refactoring specialist. Analyze the provided code and \
improve its quality without changing its behavior.

Your task is to:
1. Identify code smells, duplication, and complexity issues
2. Extract common functionality into reusable components
3. Improve naming, structure, and readability
4. Simplify complex logic and reduce cognitive load
5. Apply design patterns where appropriate
6. Maintain exact same functionality and behavior

Focus on these refactoring techniques:
- Extract methods/functions from long procedures
- Rename variables and functions for clarity
- Remove code duplication
- Simplify conditional logic
- Improve error handling
- Optimize imports and dependencies
- Enhance code organization and structure

Always preserve:
- Original functionality and behavior
- Public API compatibility
- Error handling behavior
- Performance characteristics (don't make it slower)

Provide clean, well-documented refactored code that is easier to understand and maintain."""

    async def _execute_oneshot(self, context: ExecutionContext) -> CommandOutput:
        """Execute refactoring using structured PydanticAI agent."""

        if not context.files:
            return CommandOutput(
                success=False, summary="No files provided for refactoring", metadata={"error": "no_files"}
            )

        # Create agent
        refactor_agent = self.build_agent(result_type=RefactoringResult)

        # Prepare context
        formatted_context = self._format_context(context)

        refactor_context = f"""{formatted_context}

Refactoring Guidelines:
- Preserve all existing functionality
- Improve code readability and maintainability
- Remove duplication and complexity
- Follow language best practices and conventions
- Ensure refactored code is well-documented
- Make minimal changes that have maximum impact"""

        try:
            logger.info(f"Refactoring {len(context.files)} files...")

            result = await refactor_agent.run(refactor_context)

            if hasattr(result, "data"):
                refactor_result: RefactoringResult = result.data

                # Convert refactored files to file changes
                file_changes = []
                for file_data in refactor_result.refactored_files:
                    file_changes.append(
                        FileChange(
                            filepath=file_data["filepath"],
                            content=file_data["content"],
                            metadata={
                                "command": "refactor",
                                "operations": [
                                    op.operation_type
                                    for op in refactor_result.operations
                                    if op.file_path == file_data["filepath"]
                                ],
                            },
                        )
                    )

                return CommandOutput(
                    file_changes=file_changes,
                    summary=refactor_result.summary,
                    metadata={
                        "operations": len(refactor_result.operations),
                        "complexity_reduction": refactor_result.estimate_complexity_reduction(),
                        "quality_improvements": refactor_result.quality_improvements,
                    },
                )
            else:
                return CommandOutput(
                    success=False,
                    summary="Failed to generate structured refactoring result",
                    metadata={"error": "invalid_response"},
                )

        except Exception as e:
            logger.error(f"Error in refactoring: {e}")
            return CommandOutput(
                success=False, summary=f"Failed to refactor code: {str(e)}", metadata={"error": str(e)}
            )


class FixCommand(Command):
    """Fix code issues and bugs using PydanticAI."""

    name: str = "fix"
    prompt_template: str = """You are an expert debugging and code fix specialist. Analyze the provided code \
and fix any issues you identify.

Your task is to:
1. Identify bugs, errors, and code issues
2. Fix syntax errors and logical problems
3. Improve error handling and edge case coverage
4. Fix performance issues and memory leaks
5. Address security vulnerabilities
6. Fix style and convention violations

Types of issues to look for:
- Syntax errors and typos
- Logic errors and incorrect algorithms
- Null pointer/undefined variable access
- Resource leaks and improper cleanup
- Race conditions and concurrency issues
- Input validation problems
- Error handling gaps
- Performance bottlenecks
- Security vulnerabilities

For each fix:
- Explain what the issue was
- Describe how the fix resolves it
- Ensure the fix doesn't break existing functionality
- Consider the impact and risk level of the change

Prioritize fixes by impact and risk level."""

    async def _execute_oneshot(self, context: ExecutionContext) -> CommandOutput:
        """Execute code fixing using structured PydanticAI agent."""

        if not context.files:
            return CommandOutput(success=False, summary="No files provided for fixing", metadata={"error": "no_files"})

        # Create agent
        fix_agent = self.build_agent(result_type=FixResult)

        # Prepare context with error information if available
        formatted_context = self._format_context(context)

        fix_context = f"""{formatted_context}

Fix Guidelines:
- Identify and fix actual bugs and issues
- Prioritize critical and high-impact fixes
- Provide clear explanations for each fix
- Consider backwards compatibility
- Test edge cases and error conditions
- Follow language best practices"""

        try:
            logger.info(f"Analyzing and fixing issues in {len(context.files)} files...")

            result = await fix_agent.run(fix_context)

            if hasattr(result, "data"):
                fix_result: FixResult = result.data

                # Convert fixes to file changes
                file_changes = []
                for fix_data in fix_result.to_file_changes():
                    file_changes.append(
                        FileChange(
                            filepath=fix_data["filepath"],
                            content=fix_data["content"],
                            metadata={"command": "fix", "fixes_applied": fix_data["fixes_applied"]},
                        )
                    )

                return CommandOutput(
                    file_changes=file_changes,
                    summary=fix_result.summary,
                    metadata={
                        "fixes_count": len(fix_result.fixes),
                        "files_affected": fix_result.files_affected,
                        "fix_types": [fix.fix_type for fix in fix_result.fixes],
                    },
                )
            else:
                return CommandOutput(
                    success=False,
                    summary="Failed to generate structured fix result",
                    metadata={"error": "invalid_response"},
                )

        except Exception as e:
            logger.error(f"Error in code fixing: {e}")
            return CommandOutput(success=False, summary=f"Failed to fix code: {str(e)}", metadata={"error": str(e)})


class ImplementCommand(Command):
    """Implement new features using PydanticAI."""

    name: str = "implement"
    prompt_template: str = """You are an expert software engineer. Implement the requested feature or \
functionality based on the requirements and existing code context.

Your task is to:
1. Understand the feature requirements and scope
2. Analyze existing code patterns and architecture
3. Design a clean implementation that fits the codebase
4. Write complete, working code for the feature
5. Include proper error handling and validation
6. Add appropriate tests if needed
7. Follow the project's coding conventions

Implementation guidelines:
- Write clean, readable, and maintainable code
- Follow existing architecture patterns
- Include proper documentation and comments
- Handle edge cases and error conditions
- Ensure backwards compatibility where needed
- Use appropriate design patterns
- Follow SOLID principles and best practices

Break down complex features into logical components and implement them step by step."""

    async def _execute_oneshot(self, context: ExecutionContext) -> CommandOutput:
        """Execute feature implementation using structured PydanticAI agent."""

        if not context.files:
            return CommandOutput(
                success=False, summary="No context provided for implementation", metadata={"error": "no_files"}
            )

        # Create agent
        implement_agent = self.build_agent(result_type=ImplementationResult)

        # Prepare context
        formatted_context = self._format_context(context)

        implement_context = f"""{formatted_context}

Implementation Guidelines:
- Follow existing code patterns and architecture
- Write complete, working code
- Include proper error handling
- Add documentation and comments
- Consider edge cases and validation
- Ensure code quality and maintainability"""

        try:
            logger.info(f"Implementing feature based on {len(context.files)} files...")

            result = await implement_agent.run(implement_context)

            if hasattr(result, "data"):
                impl_result: ImplementationResult = result.data

                # Convert implementation steps to file changes
                file_changes = []
                ordered_steps = impl_result.get_implementation_order()

                for step in ordered_steps:
                    file_changes.append(
                        FileChange(
                            filepath=step.file_path,
                            content=step.code,
                            metadata={
                                "command": "implement",
                                "step_name": step.step_name,
                                "step_type": step.step_type,
                                "description": step.description,
                            },
                        )
                    )

                return CommandOutput(
                    file_changes=file_changes,
                    summary=impl_result.summary,
                    metadata={
                        "steps": len(impl_result.implementation_steps),
                        "files_to_create": impl_result.files_to_create,
                        "files_to_modify": impl_result.files_to_modify,
                        "complexity": impl_result.estimated_complexity,
                    },
                )
            else:
                return CommandOutput(
                    success=False,
                    summary="Failed to generate structured implementation result",
                    metadata={"error": "invalid_response"},
                )

        except Exception as e:
            logger.error(f"Error in implementation: {e}")
            return CommandOutput(
                success=False, summary=f"Failed to implement feature: {str(e)}", metadata={"error": str(e)}
            )


class DebugTestsCommand(Command):
    """Debug failing tests using error output."""

    name: str = "debug-tests"
    prompt_template: str = """You are an expert test debugging specialist. Analyze failing tests and \
fix the issues causing them to fail.

Your task is to:
1. Parse and understand test failure output
2. Identify the root cause of each failure
3. Determine whether the issue is in the implementation or the test
4. Fix the implementation code (preferred) or test code as needed
5. Ensure fixes don't break other functionality
6. Provide clear explanations for each fix

Analysis approach:
- Read the error messages and stack traces carefully
- Understand what the test is trying to verify
- Trace through the code to find the actual issue
- Consider edge cases and boundary conditions
- Look for logical errors, typos, and assumption failures
- Check for environmental or dependency issues

Fixing strategy:
- Prefer fixing implementation over changing tests
- Only modify tests if they are incorrect or poorly written
- Ensure fixes address root causes, not just symptoms
- Validate that fixes don't introduce new issues
- Maintain test coverage and quality

Provide detailed analysis and confident fixes."""

    async def _execute_oneshot(self, context: ExecutionContext) -> CommandOutput:
        """Execute test debugging using structured PydanticAI agent."""

        if not context.clipboard:
            return CommandOutput(
                success=False,
                summary="No test failure output provided in clipboard",
                metadata={"error": "no_test_output"},
            )

        # Create agent
        debug_agent = self.build_agent(result_type=DebugResult)

        # Prepare context with emphasis on test output
        formatted_context = self._format_context(context)

        debug_context = f"""{formatted_context}

Focus on the error output in the clipboard section above. This contains the failing test information.

Debug Analysis Instructions:
- Parse all test failures from the output
- Identify the specific errors and their causes
- Trace through the code to understand the issue
- Determine the best fix for each failure
- Explain your reasoning for each fix
- Prioritize implementation fixes over test changes"""

        try:
            logger.info("Debugging test failures from output...")

            result = await debug_agent.run(debug_context)

            if hasattr(result, "data"):
                debug_result: DebugResult = result.data

                # Convert debug fixes to file changes
                file_changes = []
                for fix in debug_result.fixes_proposed:
                    file_changes.append(
                        FileChange(
                            filepath=fix.target_file,
                            content=fix.fixed_code,
                            metadata={
                                "command": "debug-tests",
                                "fix_description": fix.fix_description,
                                "confidence": fix.confidence,
                                "affects_tests": fix.affects_tests,
                            },
                        )
                    )

                return CommandOutput(
                    file_changes=file_changes,
                    summary=f"Fixed {len(debug_result.failures_analyzed)} test failures with "
                    + f"{len(debug_result.fixes_proposed)} fixes",
                    metadata={
                        "failures_analyzed": len(debug_result.failures_analyzed),
                        "fixes_proposed": len(debug_result.fixes_proposed),
                        "implementation_fixes": len(debug_result.get_implementation_fixes()),
                        "test_fixes": len(debug_result.get_test_fixes()),
                        "root_cause": debug_result.root_cause_analysis,
                        "confidence": debug_result.confidence_level,
                    },
                )
            else:
                return CommandOutput(
                    success=False,
                    summary="Failed to generate structured debug result",
                    metadata={"error": "invalid_response"},
                )

        except Exception as e:
            logger.error(f"Error in test debugging: {e}")
            return CommandOutput(success=False, summary=f"Failed to debug tests: {str(e)}", metadata={"error": str(e)})
