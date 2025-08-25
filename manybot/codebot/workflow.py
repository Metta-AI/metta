"""
Core workflow foundation for AI-powered development assistance.

Provides the base classes and patterns for structured agent operations using PydanticAI.
"""

import html
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field

from .codeclip import get_context

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
            content, token_info = get_context(paths=[Path(p) for p in paths] if paths else None)

            prompt_context = PromptContext(
                role_prompt=role_prompt,
                task_prompt=task_prompt,
                files=self._parse_files_from_content(content),
                token_count=token_info["total_tokens"],
                working_directory=Path.cwd(),
                git_diff=self._get_git_diff(),
                clipboard=self._get_clipboard(),
            )

            return prompt_context, execution_context

        except ImportError as e:
            logger.warning(f"Could not import codeclip: {e}")
            prompt_context = PromptContext(
                role_prompt=role_prompt, task_prompt=task_prompt, files={}, token_count=0, working_directory=Path.cwd()
            )
            return prompt_context, execution_context

    def _parse_files_from_content(self, xml_text: str) -> Dict[str, str]:
        """Parse files from codeclip XML-like output with some resilience."""
        files: Dict[str, str] = {}
        if not xml_text or not xml_text.strip():
            return files

        # Prefer attribute-based form: <file path="..."><content>...</content></file>
        file_pat = re.compile(
            r"<(?:file|document)\b([^>]*)>(.*?)</(?:file|document)>",
            re.DOTALL | re.IGNORECASE,
        )
        attr_path_pat = re.compile(r'\bpath="([^"]+)"', re.IGNORECASE)
        tag_pat = re.compile(
            r"<(?P<tag>source|name|document_content|instructions|content)\b[^>]*>(?P<body>.*?)</\1>",
            re.DOTALL | re.IGNORECASE,
        )

        for outer_attrs, inner in file_pat.findall(xml_text):
            # Path from attribute or from a <source>/<name> child
            m_path = attr_path_pat.search(outer_attrs)
            path = m_path.group(1) if m_path else None

            # Gather child nodes
            tags = {m.group("tag").lower(): m.group("body") for m in tag_pat.finditer(inner)}
            if not path:
                path = tags.get("source") or tags.get("name")
            if not path:
                continue

            # Choose body
            body = tags.get("document_content") or tags.get("content") or tags.get("instructions") or ""
            # Unescape XML entities
            body = html.unescape(body)

            norm_path = str(PurePosixPath(path.strip()))
            files[norm_path] = body

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
