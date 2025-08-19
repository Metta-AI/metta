"""
Core workflow foundation for AI-powered development assistance.

Provides the base classes and patterns for structured agent operations using PydanticAI.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union, Literal
import logging

from pydantic import BaseModel, Field, validator
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


class FileChange(BaseModel):
    """Atomic unit of code modification"""

    filepath: str
    content: str
    operation: Literal["write", "delete"] = "write"

    def apply(self) -> None:
        """Apply change to filesystem"""
        path = Path(self.filepath)

        if self.operation == "delete":
            if path.exists():
                path.unlink()
        else:  # write
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.content, encoding="utf-8")

    def preview(self) -> str:
        """Generate diff preview"""
        if self.operation == "delete":
            return f"DELETE: {self.filepath}"
        else:
            lines = self.content.split("\n")
            preview_lines = lines[:10]
            if len(lines) > 10:
                preview_lines.append(f"... ({len(lines) - 10} more lines)")
            return f"WRITE: {self.filepath}\n" + "\n".join(f"+ {line}" for line in preview_lines)


class ExecutionContext(BaseModel):
    """Context passed to commands"""

    git_diff: str = ""
    clipboard: str = ""
    files: Dict[str, str] = {}
    working_directory: Path
    token_count: int = 0


class CommandOutput(BaseModel):
    """Standard output from any command"""

    file_changes: List[FileChange] = Field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


T = TypeVar("T", bound=BaseModel)


class Command(BaseModel):
    """Base class for AI-powered commands using PydanticAI agents"""

    name: str
    prompt_template: str
    default_paths: List[str] = Field(default_factory=list)
    result_type: type[BaseModel] = CommandOutput

    def build_agent(self) -> Agent[T]:
        """Build PydanticAI agent for this command"""
        return Agent(result_type=self.result_type, system_prompt=self.prompt_template)

    async def execute(self, context: ExecutionContext, mode: Optional[str] = None) -> CommandOutput:
        """Execute command using appropriate mode"""

        # Build agent
        agent = self.build_agent()

        # Prepare context
        agent_context = self._format_context(context)

        # Execute based on mode
        if mode == "interactive":
            return await self._execute_interactive(agent_context)
        elif mode == "claudesdk":
            return await self._execute_claudesdk(agent, agent_context)
        else:
            # Default oneshot execution with PydanticAI
            result = await agent.run(agent_context)
            return self._process_result(result.data)

    def _format_context(self, context: ExecutionContext) -> Dict[str, Any]:
        """Format context for agent consumption"""
        formatted = {"files": context.files, "working_directory": str(context.working_directory)}

        if context.git_diff:
            formatted["git_diff"] = context.git_diff

        if context.clipboard:
            formatted["error_output"] = context.clipboard

        return formatted

    def _process_result(self, result: Any) -> CommandOutput:
        """Process agent result into CommandOutput"""
        if isinstance(result, CommandOutput):
            return result

        # Default implementation - subclasses should override for custom result types
        return CommandOutput(summary=str(result), metadata={"raw_result": result})

    async def _execute_interactive(self, agent_context: Dict[str, Any]) -> CommandOutput:
        """Execute in interactive mode (placeholder)"""
        # TODO: Implement interactive mode
        logger.warning("Interactive mode not yet implemented, falling back to oneshot")
        agent = self.build_agent()
        result = await agent.run(agent_context)
        return self._process_result(result.data)

    async def _execute_claudesdk(self, agent: Agent, agent_context: Dict[str, Any]) -> CommandOutput:
        """Execute via Claude SDK mode (placeholder)"""
        # TODO: Implement Claude SDK mode
        logger.warning("Claude SDK mode not yet implemented, falling back to oneshot")
        result = await agent.run(agent_context)
        return self._process_result(result.data)


class ContextManager:
    """Smart context gathering using codeclip"""

    def gather_context(self, paths: List[str]) -> ExecutionContext:
        """Gather files using codeclip with intelligent prioritization:

        1. Git diff files (10x weight)
        2. Test files for source (2x weight)
        3. Import relationships (1.5x weight)
        4. Recently modified (1.2x weight)
        5. Smaller files when equal priority
        """
        try:
            # Import here to avoid circular dependencies
            from .codeclip.codeclip.file import get_context

            content, token_info = get_context(paths=paths)

            return ExecutionContext(
                files=self._parse_files_from_content(content),
                token_count=token_info["total_tokens"],
                working_directory=Path.cwd(),
                git_diff=self._get_git_diff(),
                clipboard=self._get_clipboard(),
            )
        except ImportError as e:
            logger.warning(f"Could not import codeclip: {e}")
            return ExecutionContext(files={}, token_count=0, working_directory=Path.cwd())

    def _parse_files_from_content(self, content: str) -> Dict[str, str]:
        """Parse files from codeclip XML output"""
        import re

        files = {}

        # Parse XML format from codeclip
        pattern = (
            r"<document .*?>\s*<source>(.*?)</source>.*?"
            r"(?:<document_content>(.*?)</document_content>|<instructions>(.*?)</instructions>).*?</document>"
        )
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            path = match[0].strip()
            content = match[1] if match[1] else match[2]
            files[path] = content

        return files

    def _get_git_diff(self) -> str:
        """Get current git diff"""
        try:
            import subprocess

            result = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True, check=False)
            return result.stdout
        except Exception:
            return ""

    def _get_clipboard(self) -> str:
        """Get clipboard content if available"""
        try:
            import pyperclip

            return pyperclip.paste()
        except Exception:
            return ""
