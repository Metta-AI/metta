"""
Summarize workflow for AI-powered code analysis.

Implements the summarizer pattern as described in the codebot README - takes a set of files
and produces a token-constrained summary optimized for AI consumption.
"""

import logging
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field, validator
from pydantic_ai import Agent, ModelRetry

from manybot.codebot.workflow import Command, CommandOutput, ExecutionContext, FileChange

logger = logging.getLogger(__name__)


# Define structured models for the summarizer
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
            result_type=SummaryResult,
            system_prompt=self.prompt_template.format(token_limit=token_limit),
        )

        # Run agent with retry logic
        try:
            result = await summarizer.run({"files": context.files})
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


class SummaryCache:
    """Cache summaries to avoid recomputation - sits on top of SummarizeCommand"""

    def __init__(self, cache_dir: Path = Path(".codebot/summaries")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.summarize_command = SummarizeCommand()

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

    async def get_or_create_summary(self, context: ExecutionContext, token_limit: int = 2000) -> CommandOutput:
        """Get cached summary or create new one using SummarizeCommand"""

        # Generate cache key from file paths + modification times + token limit
        cache_key = self._generate_cache_key(list(context.files.keys()), token_limit)
        cache_path = self.cache_dir / f"{cache_key}.md"

        if cache_path.exists():
            # Return cached summary as CommandOutput
            cached_content = cache_path.read_text()
            return CommandOutput(
                file_changes=[FileChange(filepath=".codebot/summaries/latest.md", content=cached_content)],
                summary=f"Retrieved cached summary for {len(context.files)} files",
                metadata={"cached": True, "cache_key": cache_key},
            )

        # Cache miss - delegate to SummarizeCommand for actual AI work
        logger.info(f"Cache miss for key {cache_key}, generating new summary")
        result = await self.summarize_command.execute(context, token_limit)

        # Cache the generated summary
        if result.file_changes:
            summary_content = result.file_changes[0].content
            cache_path.write_text(summary_content)
            logger.info(f"Cached summary with key {cache_key}")

        return result
