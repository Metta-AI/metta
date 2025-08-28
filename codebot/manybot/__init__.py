"""
Codebot: AI-powered development assistance through a unified CLI.

Provides structured AI operations with PydanticAI for code analysis,
testing, refactoring, and more.
"""

from . import cli
from .commands import SummarizeCommand, run_summarize_command
from .models import CommandOutput, ExecutionContext, FileChange

__version__ = "0.1.0"
__all__ = ["FileChange", "ExecutionContext", "CommandOutput", "SummarizeCommand", "run_summarize_command", "cli"]
