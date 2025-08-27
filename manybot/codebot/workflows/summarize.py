"""
Summarize workflow for AI-powered code analysis.

Implements the summarizer pattern as described in the codebot README - takes a set of files
and produces a token-constrained summary optimized for AI consumption.
"""

from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, ModelRetry

from manybot.codebot.logging import get_logger
from manybot.codebot.workflow import Command, CommandOutput, ExecutionContext, FileChange, PromptContext

logger = get_logger(__name__)


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
