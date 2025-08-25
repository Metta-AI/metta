"""
Summarize workflow for AI-powered code analysis.

Implements the summarizer pattern as described in the codebot README - takes a set of files
and produces a token-constrained summary optimized for AI consumption.
"""

import logging
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry

from manybot.codebot.workflow import Command, CommandOutput, ExecutionContext, FileChange, PromptContext

logger = logging.getLogger(__name__)


class SummaryResult(BaseModel):
    """
    V0: summarizer returns one markdown document.
    We keep to_markdown() so existing execute() code remains unchanged.
    """

    model_config = ConfigDict(extra="ignore")

    content: str = Field(description="Complete markdown content of the summary")

    # Backward-compat – execute() calls this today.
    def to_markdown(self) -> str:
        return self.content


class SummarizeCommand(Command):
    """Implementation using PydanticAI agents with role + task pattern"""

    name: str = "summarize"

    async def execute(
        self, prompt_context: PromptContext, execution_context: ExecutionContext, token_limit: int = 10000
    ) -> CommandOutput:
        """Execute summarization using structured PydanticAI agent"""

        # Create agent with role from context
        summarizer = Agent(
            result_type=SummaryResult,
            system_prompt=prompt_context.role_prompt,
        )

        # Format task instructions with parameters
        formatted_task = prompt_context.task_prompt.format(token_limit=token_limit)

        # Run agent with retry logic
        try:
            if execution_context.mode == "oneshot":
                result = await summarizer.run(formatted_task, files=prompt_context.files)
            else:
                # Default oneshot execution with PydanticAI
                logger.warning(f"{execution_context.mode} mode not yet implemented, falling back to oneshot")
                result = await summarizer.run(formatted_task, files=prompt_context.files)

            # Convert to markdown
            summary_content = result.data.to_markdown()

            # Create output file
            return CommandOutput(
                file_changes=[FileChange(filepath=".codebot/summaries/latest.md", content=summary_content)],
                summary=f"Analyzed {len(prompt_context.files)} files and generated summary",
                metadata={
                    "file_count": len(prompt_context.files),
                    "content_length": len(summary_content),
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

    async def get_or_create_summary(
        self, prompt_context: PromptContext, execution_context: ExecutionContext, token_limit: int = 10000
    ) -> CommandOutput:
        """Get cached summary or create new one using SummarizeCommand"""

        # Generate cache key from file paths + modification times + token limit
        cache_key = self._generate_cache_key(list(prompt_context.files.keys()), token_limit)
        cache_path = self.cache_dir / f"{cache_key}.md"

        if cache_path.exists():
            # Return cached summary as CommandOutput
            cached_content = cache_path.read_text()
            return CommandOutput(
                file_changes=[FileChange(filepath=".codebot/summaries/latest.md", content=cached_content)],
                metadata={"cached": True, "cache_key": cache_key},
            )

        # Cache miss - delegate to SummarizeCommand for actual AI work
        logger.info(f"Cache miss for key {cache_key}, generating new summary")
        result = await self.summarize_command.execute(prompt_context, execution_context, token_limit)

        # Cache the generated summary
        if result.file_changes:
            summary_content = result.file_changes[0].content
            cache_path.write_text(summary_content)
            logger.info(f"Cached summary with key {cache_key}")

        return result
