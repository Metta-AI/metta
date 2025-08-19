"""
Summarize workflow for AI-powered code analysis.

Implements the summarizer pattern as described in the codebot README - takes a set of files
and produces a token-constrained summary optimized for AI consumption.
"""

from pathlib import Path
from typing import List, Dict, Optional, Literal
import logging

from pydantic import BaseModel, Field, validator
from pydantic_ai import Agent, ModelRetry

from ..workflow import Command, CommandOutput, ExecutionContext, FileChange

logger = logging.getLogger(__name__)


class CodeComponent(BaseModel):
    """A significant code component identified in the summary"""

    name: str
    type: Literal["class", "function", "module", "interface", "service"]
    description: str
    file_path: str
    dependencies: List[str] = Field(default_factory=list)


class CodePattern(BaseModel):
    """Identified pattern or convention in the codebase"""

    pattern: str
    description: str
    examples: List[str] = Field(default_factory=list)


class SummaryResult(BaseModel):
    """Structured output from code summarization"""

    overview: str = Field(description="High-level overview of the codebase")
    components: List[CodeComponent] = Field(description="Key components identified")
    external_dependencies: List[str] = Field(description="External packages/libraries used")
    patterns: List[CodePattern] = Field(description="Common patterns and conventions")
    entry_points: List[str] = Field(description="Main entry points into the code")

    @validator("overview")
    def overview_not_too_long(cls, v):
        # Ensure overview stays concise
        if len(v.split()) > 500:
            raise ValueError("Overview must be under 500 words")
        return v

    def to_markdown(self) -> str:
        """Convert to readable markdown format"""
        sections = [
            f"# Code Summary\n\n{self.overview}",
            "\n## Key Components\n" + "\n".join(f"- **{c.name}** ({c.type}): {c.description}" for c in self.components),
            "\n## Dependencies\n" + "\n".join(f"- {d}" for d in self.external_dependencies),
            "\n## Patterns\n" + "\n".join(f"- **{p.pattern}**: {p.description}" for p in self.patterns),
        ]
        return "\n".join(sections)


class SummarizeCommand(Command):
    """Implementation using PydanticAI agents"""

    name: str = "summarize"
    prompt_template: str = """Analyze code to create a structured summary optimized for AI consumption.
    Focus on architecture, key components, and patterns that would help another AI understand the codebase quickly.
    Keep the total summary under {token_limit} tokens."""
    result_type: type[BaseModel] = SummaryResult

    async def execute(self, context: ExecutionContext, token_limit: int = 2000) -> CommandOutput:
        """Execute summarization using structured PydanticAI agent"""

        # Create agent with structured result type
        summarizer = Agent(
            "claude-3-5-sonnet-20241022",
            result_type=SummaryResult,
            system_prompt=self.prompt_template.format(token_limit=token_limit),
        )

        # Prepare context
        summary_context = {"files": context.files, "focus_areas": self._identify_focus_areas(context)}

        # Run agent with retry logic
        try:
            result = await summarizer.run(summary_context)
            summary = result.data

            # Convert to markdown
            summary_content = summary.to_markdown()

            # Create output file
            return CommandOutput(
                file_changes=[FileChange(filepath=".codebot/summaries/latest.md", content=summary_content)],
                summary=f"Analyzed {len(context.files)} files, found {len(summary.components)} key components",
                metadata={
                    "component_count": len(summary.components),
                    "dependency_count": len(summary.external_dependencies),
                    "pattern_count": len(summary.patterns),
                },
            )

        except ModelRetry as retry:
            # Handle retry with additional context
            logger.warning(f"Retrying summary: {retry.message}")
            raise

    def _identify_focus_areas(self, context: ExecutionContext) -> List[str]:
        """Identify areas to focus analysis on based on file types and patterns"""
        focus_areas = []

        # Analyze file extensions
        extensions = set()
        for filepath in context.files.keys():
            ext = Path(filepath).suffix.lower()
            if ext:
                extensions.add(ext)

        # Suggest focus areas based on extensions
        if ".py" in extensions:
            focus_areas.append("Python architecture and class hierarchies")
        if ".js" in extensions or ".ts" in extensions:
            focus_areas.append("JavaScript/TypeScript module structure")
        if ".java" in extensions:
            focus_areas.append("Java package organization and design patterns")
        if ".rs" in extensions:
            focus_areas.append("Rust module system and ownership patterns")

        # Look for configuration files
        config_files = [
            f
            for f in context.files.keys()
            if any(
                config_name in Path(f).name.lower()
                for config_name in ["config", "settings", "package.json", "cargo.toml", "pyproject.toml"]
            )
        ]
        if config_files:
            focus_areas.append("Configuration and dependency management")

        # Look for test files
        test_files = [
            f
            for f in context.files.keys()
            if any(test_indicator in f.lower() for test_indicator in ["test", "spec", "__test__"])
        ]
        if test_files:
            focus_areas.append("Testing patterns and coverage")

        return focus_areas or ["General code structure and patterns"]


class SummaryCache:
    """Cache summaries to avoid recomputation"""

    def __init__(self, cache_dir: Path = Path(".codebot/summaries")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, files: List[str], token_limit: int) -> str:
        """Generate cache key from file paths + modification times + token limit"""
        import hashlib

        # Collect file info
        file_info = []
        for filepath in sorted(files):
            try:
                path = Path(filepath)
                if path.exists():
                    mtime = path.stat().st_mtime
                    file_info.append(f"{filepath}:{mtime}")
                else:
                    file_info.append(f"{filepath}:missing")
            except Exception:
                file_info.append(f"{filepath}:error")

        # Include token limit in cache key
        cache_input = f"{token_limit}:" + "|".join(file_info)
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _gather_file_contents(self, files: List[str]) -> Dict[str, str]:
        """Gather file contents for caching"""
        contents = {}
        for filepath in files:
            try:
                path = Path(filepath)
                if path.exists() and path.is_file():
                    contents[filepath] = path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read {filepath}: {e}")
        return contents

    async def get_or_create_summary(self, files: List[str], token_limit: int = 2000) -> str:
        """Get cached summary or create new one"""

        # Generate cache key from file paths + modification times + token limit
        cache_key = self._generate_cache_key(files, token_limit)
        cache_path = self.cache_dir / f"{cache_key}.md"

        if cache_path.exists():
            return cache_path.read_text()

        # Create new summary using PydanticAI
        summarizer = Agent(
            "claude-3-5-sonnet-20241022",
            result_type=SummaryResult,
            system_prompt=f"""Create a summary of the provided code that:
            1. Captures the essential functionality and structure
            2. Preserves important technical details
            3. Stays under {token_limit} tokens
            4. Is optimized for another LLM to quickly understand the codebase""",
        )

        result = await summarizer.run({"files": self._gather_file_contents(files), "token_limit": token_limit})

        summary = result.data.to_markdown()
        cache_path.write_text(summary)

        return summary
