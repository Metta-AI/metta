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

        # Format task instructions with parameters
        formatted_task = prompt_context.task_prompt.format(token_limit=token_limit)

        # Format file contents for the prompt
        file_contents = "\n\n".join([
            f"## File: {filepath}\n\n```\n{content}\n```"
            for filepath, content in prompt_context.files.items()
        ])

        # Combine task and file contents into a single prompt
        full_prompt = f"{formatted_task}\n\n# Files to Analyze\n\n{file_contents}"

        # Handle dry-run mode
        if execution_context.dry_run:
            # For dry-run, create a mock summary without calling the AI
            summary_content = f"# Code Summary (DRY RUN)\n\nWould analyze {len(prompt_context.files)} files:\n\n" + \
                            "\n".join([f"- {filepath}" for filepath in prompt_context.files.keys()]) + \
                            f"\n\nPrompt length: {len(full_prompt)} characters\nToken limit: {token_limit}"
        else:
            # Create agent with role from context only when not in dry-run mode
            # Use Claude as default model if no real model is configured
            model_name = execution_context.model or "anthropic:claude-3-5-sonnet-20241022"
            summarizer = Agent(
                model=model_name,
                output_type=SummaryResult,
                system_prompt=prompt_context.role_prompt,
            )
            # Run agent with retry logic
            try:
                if execution_context.mode == "oneshot":
                    result = await summarizer.run(full_prompt)
                else:
                    # Default oneshot execution with PydanticAI
                    logger.warning(f"{execution_context.mode} mode not yet implemented, falling back to oneshot")
                    result = await summarizer.run(full_prompt)

                # Convert to markdown
                summary_content = result.output.to_markdown()
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                raise

        # Create output file
        return CommandOutput(
            file_changes=[FileChange(filepath=".codebot/summaries/latest.md", content=summary_content)],
            summary=f"Analyzed {len(prompt_context.files)} files and generated summary",
            metadata={
                "file_count": len(prompt_context.files),
                "content_length": len(summary_content),
            },
        )
