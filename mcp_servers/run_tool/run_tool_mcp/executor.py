"""Executor for running tools programmatically using existing utilities."""

import io
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from metta.common.tool.recipe_registry import recipe_registry
from metta.common.tool.run_tool import build_and_execute_tool, nestify
from metta.common.tool.tool_path import parse_two_token_syntax, resolve_and_load_tool_maker
from metta.common.tool.tool_registry import tool_registry

from .models import ErrorResponse, ToolExecutionResult
from .tools import get_tool_arguments
from .utils import determine_error_type, format_command_preview

logger = logging.getLogger(__name__)


def format_execution_summary(command: str, dry_run: bool) -> str:
    """Format execution summary message."""
    if dry_run:
        return f"Would execute: {command} (dry run - validation only)"
    return f"Executed: {command}"


def create_execution_result(
    success: bool,
    exit_code: int,
    stdout: str,
    stderr: str,
    command: str,
    error: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Create ToolExecutionResult and return as dict."""
    result = ToolExecutionResult(
        success=success,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        command=command,
        error=error,
        summary=summary,
    )
    return result.model_dump()


class RunToolExecutor:
    """Executes tools programmatically using existing run_tool utilities."""

    def __init__(self, run_script_path: Path, repo_root: Path, timeout: int = 3600):
        self.run_script_path = run_script_path
        self.repo_root = repo_root
        self.timeout = timeout

    async def execute(
        self,
        tool_path: str,
        arguments: Optional[dict[str, Any]] = None,
        dry_run: bool = False,
        verbose: bool = False,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """Execute a tool programmatically."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            resolved_tool_path, _ = parse_two_token_syntax(tool_path, None)
            cli_args = arguments or {}
            nested_cli = nestify(cli_args)
            equivalent_command = format_command_preview(resolved_tool_path, cli_args)

            tool_maker = resolve_and_load_tool_maker(resolved_tool_path)
            if tool_maker is None:
                return create_execution_result(
                    success=False,
                    exit_code=1,
                    stdout=stdout_capture.getvalue(),
                    stderr=f"Could not find tool '{tool_path}'",
                    command=equivalent_command,
                    error="tool_not_found",
                    summary=f"Tool '{tool_path}' not found",
                )

            if verbose:
                logger.info(f"Loading tool: {tool_maker.__module__}.{tool_maker.__name__}")

            def output_info_handler(msg: str) -> None:
                if verbose:
                    logger.info(msg)
                stdout_capture.write(msg + "\n")

            def output_error_handler(msg: str) -> None:
                logger.error(msg)
                stderr_capture.write(msg + "\n")

            def output_exception_handler(msg: str) -> None:
                logger.exception(msg)
                stderr_capture.write(msg + "\n")

            exit_code = build_and_execute_tool(
                tool_maker=tool_maker,
                nested_cli=nested_cli,
                cli_args=cli_args,
                tool_path=tool_path,
                dry_run=dry_run,
                verbose=verbose,
                output_info_func=output_info_handler,
                output_error_func=output_error_handler,
                output_exception_func=output_exception_handler,
            )

            stderr_content = stderr_capture.getvalue()
            error_type = determine_error_type(exit_code, stderr_content)
            summary = format_execution_summary(equivalent_command, dry_run)

            return create_execution_result(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_content,
                command=equivalent_command,
                error=error_type,
                summary=summary,
            )

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    async def execute_list_command(
        self, recipe: Optional[str] = None, tool_type: Optional[str] = None
    ) -> dict[str, Any]:
        """Execute a --list command to discover tools using recipe_registry."""
        if recipe:
            recipe_obj = recipe_registry.get(recipe)
            if recipe_obj:
                tools = sorted(recipe_obj.get_all_tool_maker_names())
                output_lines = [f"  {recipe_obj.short_name}.{tool}" for tool in tools]
                return create_execution_result(
                    success=True,
                    exit_code=0,
                    stdout="\n".join(output_lines),
                    stderr="",
                    command=f"./tools/run.py {recipe} --list",
                )
            return create_execution_result(
                success=False,
                exit_code=1,
                stdout="",
                stderr=f"Recipe '{recipe}' not found",
                command=f"./tools/run.py {recipe} --list",
            )

        elif tool_type:
            if tool_type not in tool_registry.name_to_tool:
                return create_execution_result(
                    success=False,
                    exit_code=1,
                    stdout="",
                    stderr=f"Tool type '{tool_type}' not found",
                    command=f"./tools/run.py {tool_type} --list",
                )

            recipes = recipe_registry.get_all()
            output_lines = []
            for r in sorted(recipes, key=lambda x: x.module_name):
                makers = r.get_makers_for_tool(tool_type)
                if makers:
                    for maker_name, _ in makers:
                        output_lines.append(f"  {r.short_name}.{maker_name}")

            if output_lines:
                return create_execution_result(
                    success=True,
                    exit_code=0,
                    stdout="\n".join(output_lines),
                    stderr="",
                    command=f"./tools/run.py {tool_type} --list",
                )
            return create_execution_result(
                success=True,
                exit_code=0,
                stdout=f"No recipes found supporting '{tool_type}'",
                stderr="",
                command=f"./tools/run.py {tool_type} --list",
            )

        return create_execution_result(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Either recipe or tool_type must be provided",
            command="./tools/run.py --list",
        )

    async def execute_help_command(self, tool_path: str) -> dict[str, Any]:
        """Get tool arguments using get_tool_arguments."""
        help_command = f"./tools/run.py {tool_path} --help"
        try:
            result_json = await get_tool_arguments(tool_path)
            try:
                error_response = ErrorResponse.model_validate_json(result_json)
                if error_response.status == "error":
                    return create_execution_result(
                        success=False,
                        exit_code=1,
                        stdout="",
                        stderr=error_response.message,
                        command=help_command,
                        error="help_error",
                        summary=f"Error getting tool arguments: {error_response.message}",
                    )
            except Exception:
                pass

            return create_execution_result(
                success=True,
                exit_code=0,
                stdout=result_json,
                stderr="",
                command=help_command,
                summary="Tool arguments retrieved successfully",
            )
        except Exception as e:
            logger.exception("Error executing help command")
            return create_execution_result(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                command=help_command,
                error="help_error",
                summary=f"Error executing help command: {str(e)}",
            )
