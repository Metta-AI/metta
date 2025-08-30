"""
Core data models for codebot operations.

These models define the fundamental data structures used throughout the system.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class FileChange(BaseModel):
    """Atomic unit of code modification."""

    filepath: str = Field(description="Path to the file to modify")
    content: str = Field(description="Content to write to the file")
    operation: Literal["write", "delete"] = Field(default="write", description="Operation to perform")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def apply(self) -> None:
        """Apply change to filesystem."""
        file_path = Path(self.filepath)

        if self.operation == "delete":
            if file_path.exists():
                file_path.unlink()
        else:  # write operation
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content to file
            file_path.write_text(self.content, encoding="utf-8")

    def preview(self) -> str:
        """Generate diff preview."""
        if self.operation == "delete":
            return f"DELETE: {self.filepath}"

        file_path = Path(self.filepath)
        if file_path.exists():
            old_content = file_path.read_text(encoding="utf-8")
            # Simple diff representation - could be enhanced with actual diff library
            return (
                f"MODIFY: {self.filepath}\n--- Original\n+++ Modified\n{self._simple_diff(old_content, self.content)}"
            )
        else:
            return f"CREATE: {self.filepath}\n{self.content[:200]}{'...' if len(self.content) > 200 else ''}"

    def _simple_diff(self, old: str, new: str) -> str:
        """Generate a simple diff representation."""
        old_lines = old.splitlines()
        new_lines = new.splitlines()

        # Simple line-by-line comparison
        diff_lines = []
        max_lines = max(len(old_lines), len(new_lines))

        for i in range(max_lines):
            old_line = old_lines[i] if i < len(old_lines) else ""
            new_line = new_lines[i] if i < len(new_lines) else ""

            if old_line != new_line:
                if old_line:
                    diff_lines.append(f"- {old_line}")
                if new_line:
                    diff_lines.append(f"+ {new_line}")

        return "\n".join(diff_lines[:10])  # Limit to first 10 differences


class ExecutionContext(BaseModel):
    """Context passed to commands."""

    git_diff: str = Field(default="", description="Git diff content")
    clipboard: str = Field(default="", description="Clipboard content (e.g., error output)")
    files: Dict[str, str] = Field(default_factory=dict, description="File paths mapped to their content")
    working_directory: Path = Field(default_factory=Path.cwd, description="Current working directory")
    token_count: int = Field(default=0, description="Total token count of context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")

    class Config:
        arbitrary_types_allowed = True


class CommandOutput(BaseModel):
    """Standard output from any command."""

    file_changes: List[FileChange] = Field(default_factory=list, description="Files to be modified")
    summary: str = Field(default="", description="Human-readable summary of the operation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional operation metadata")
    success: bool = Field(default=True, description="Whether the operation completed successfully")

    def apply_changes(self) -> List[str]:
        """Apply all file changes and return list of modified files."""
        modified_files = []

        for change in self.file_changes:
            try:
                change.apply()
                modified_files.append(change.filepath)
            except Exception as e:
                # Log error but continue with other changes
                print(f"Error applying change to {change.filepath}: {e}")

        return modified_files

    def preview_changes(self) -> str:
        """Generate preview of all changes."""
        if not self.file_changes:
            return "No file changes to preview."

        previews = []
        for change in self.file_changes:
            previews.append(change.preview())

        return "\n" + "=" * 50 + "\n".join(previews)
