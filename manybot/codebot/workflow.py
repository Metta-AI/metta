"""
Core workflow foundation for AI-powered development assistance.

Provides the base classes and patterns for structured agent operations using PydanticAI.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeVar

import pyperclip
from pydantic import BaseModel, Field

from .codeclip import get_context_objects
from .logging import get_logger

logger = get_logger(__name__)


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


class PromptContext(BaseModel):
    """Context passed to commands"""

    role_prompt: str = ""
    task_prompt: str = ""
    git_diff: str = ""
    clipboard: str = ""
    files: Dict[str, str] = Field(default_factory=dict)
    working_directory: Path
    token_count: int = 0


class ExecutionContext(BaseModel):
    """Context passed to commands"""

    mode: Literal["oneshot", "claudesdk", "interactive"] = "oneshot"
    # Optional future knobs:
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    retries: int = 0
    dry_run: bool = False


class CommandOutput(BaseModel):
    """Standard output from any command"""

    file_changes: List[FileChange] = Field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


T = TypeVar("T", bound=BaseModel)


class Command(BaseModel):
    """Base class for AI-powered commands using PydanticAI agents with role + task pattern"""

    name: str
    default_paths: List[str] = Field(default_factory=list)

    async def execute(self, prompt_context: PromptContext, execution_context: ExecutionContext) -> CommandOutput:
        """Execute command using appropriate mode - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute method")


class ContextManager:
    """Smart context gathering using codeclip"""

    def _resolve_prompt_path(self, name_or_path: str, prompt_type: str) -> str:
        """Resolve simple name or path to full path under prompts/"""
        # If it already contains a path separator, treat as relative path
        if "/" in name_or_path:
            return name_or_path

        # Otherwise, treat as simple name and add .md extension if needed
        name = name_or_path if name_or_path.endswith(".md") else f"{name_or_path}.md"
        return f"{prompt_type}/{name}"

    def _load_prompt_file(self, prompt_file: str, fallback: str) -> str:
        """Load prompt from file with fallback"""
        try:
            prompt_path = Path(__file__).parent / "prompts" / prompt_file
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"Prompt file {prompt_file} not found, using fallback")
            return fallback

    def gather_context(
        self,
        paths: List[str],
        role: str,
        task: str,
        mode: Literal["oneshot", "claudesdk", "interactive"] = "oneshot",
        dry_run: bool = False,
        **execution_kwargs,
    ) -> tuple[PromptContext, ExecutionContext]:
        """
        Gather files using codeclip with intelligent prioritization and create execution context
        Returns: Tuple of (PromptContext, ExecutionContext)
        """

        # Resolve and load role and task prompts
        role_path = self._resolve_prompt_path(role, "roles")
        task_path = self._resolve_prompt_path(task, "tasks")

        role_prompt = self._load_prompt_file(
            role_path, "You are a senior software engineer with expertise in code analysis and documentation."
        )
        task_prompt = self._load_prompt_file(task_path, "Analyze the provided code and create a structured summary.")

        # Create execution context
        execution_context = ExecutionContext(mode=mode, dry_run=dry_run, **execution_kwargs)

        try:
            ctx = get_context_objects(paths=[Path(p) for p in paths] if paths else None)

            prompt_context = PromptContext(
                role_prompt=role_prompt,
                task_prompt=task_prompt,
                files=ctx.files,
                token_count=ctx.total_tokens,
                working_directory=Path.cwd(),
                git_diff=self._get_git_diff(),
                clipboard=self._get_clipboard_or_empty(),
            )

            return prompt_context, execution_context

        except ImportError as e:
            logger.warning(f"Could not import codeclip: {e}")
            prompt_context = PromptContext(
                role_prompt=role_prompt, task_prompt=task_prompt, files={}, token_count=0, working_directory=Path.cwd()
            )
            return prompt_context, execution_context

    def _get_git_diff(self) -> str:
        """Get current git diff"""
        try:
            import subprocess

            result = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True, check=False)
            return result.stdout
        except Exception:
            return ""

    def _get_clipboard_or_empty(self) -> str:
        """Get clipboard content if available"""
        if pyperclip is not None:
            return pyperclip.paste()

        return ""
