"""Executor for running tools programmatically using existing utilities."""

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from metta.common.tool.recipe_registry import recipe_registry
from metta.common.tool.run_tool import build_and_execute_tool, nestify, parse_cli_args
from metta.common.tool.tool_path import parse_two_token_syntax, resolve_and_load_tool_maker
from metta.common.tool.tool_registry import tool_registry

from .models import ErrorResponse, ToolExecutionResult
from .tools.run_tool import get_tool_arguments

logger = logging.getLogger(__name__)


class RunToolExecutor:
    """Executes tools programmatically using existing run_tool utilities.

    Uses the same logic as run_tool.main() but executes directly without subprocesses.
    """

    def __init__(self, run_script_path: Path, repo_root: Path, timeout: int = 3600):
        """Initialize the executor.

        Args:
            run_script_path: Path to the run.py script (kept for compatibility, not used)
            repo_root: Root of the Metta repository
            timeout: Default timeout in seconds (for future async support)
        """
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
        """Execute a tool programmatically.

        Replicates run_tool.main() logic but executes directly without subprocess.

        Args:
            tool_path: Tool path (e.g., 'train arena', 'arena.train')
            arguments: Dictionary of key=value arguments
            dry_run: If True, validate without executing
            verbose: If True, show verbose output
            timeout: Override default timeout

        Returns:
            Dictionary with success, exit_code, stdout, stderr, error, command
        """
        # Capture stdout/stderr to collect tool output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            cli_args_list: list[str] = []
            if arguments:
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse arguments as JSON: {arguments}")
                        arguments = None
                if isinstance(arguments, dict):
                    for key, value in arguments.items():
                        if value is None:
                            continue
                        if isinstance(value, bool):
                            value_str = str(value).lower()
                        elif isinstance(value, (list, dict)):
                            value_str = json.dumps(value)
                        else:
                            value_str = str(value)
                        cli_args_list.append(f"{key}={value_str}")

            second_token = None
            if cli_args_list and "=" not in cli_args_list[0] and not cli_args_list[0].startswith("-"):
                second_token = cli_args_list[0]

            resolved_tool_path, args_consumed = parse_two_token_syntax(tool_path, second_token)

            remaining_cli_args = cli_args_list[args_consumed:]

            cmd_parts = ["./tools/run.py", resolved_tool_path]
            if remaining_cli_args:
                cmd_parts.extend(remaining_cli_args)
            equivalent_command = " ".join(cmd_parts)

            try:
                cli_args = parse_cli_args(remaining_cli_args)
            except ValueError as e:
                result = ToolExecutionResult(
                    success=False,
                    exit_code=2,
                    stdout=stdout_capture.getvalue(),
                    stderr=f"Error parsing arguments: {e}",
                    command=equivalent_command,
                    error="argument_parse_error",
                    summary=f"Failed to parse arguments: {e}",
                )
                return result.model_dump()

            nested_cli = nestify(cli_args)

            tool_maker = resolve_and_load_tool_maker(resolved_tool_path)
            if tool_maker is None:
                result = ToolExecutionResult(
                    success=False,
                    exit_code=1,
                    stdout=stdout_capture.getvalue(),
                    stderr=f"Could not find tool '{tool_path}'",
                    command=equivalent_command,
                    error="tool_not_found",
                    summary=f"Tool '{tool_path}' not found",
                )
                return result.model_dump()

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

            error_type = None
            if exit_code != 0:
                stderr_content = stderr_capture.getvalue()
                if "Unknown arguments" in stderr_content:
                    error_type = "unknown_arguments"
                elif "Error creating tool configuration" in stderr_content:
                    error_type = "tool_construction_error"
                elif "Error applying override" in stderr_content:
                    error_type = "override_error"
                elif exit_code == 130:
                    error_type = "interrupted"
                elif "Tool invocation failed" in stderr_content:
                    error_type = "invocation_error"
                else:
                    error_type = "execution_error"

            summary = None
            if dry_run:
                summary = f"Would execute: {equivalent_command} (dry run - validation only)"
            else:
                summary = f"Executed: {equivalent_command}"

            result = ToolExecutionResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                command=equivalent_command,
                error=error_type,
                summary=summary,
            )

            return result.model_dump()

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    async def execute_list_command(
        self, recipe: Optional[str] = None, tool_type: Optional[str] = None
    ) -> dict[str, Any]:
        """Execute a --list command to discover tools using recipe_registry.

        Args:
            recipe: Recipe name to list tools for (e.g., 'arena')
            tool_type: Tool type to find recipes for (e.g., 'train')

        Returns:
            Dictionary with discovery results
        """
        if recipe:
            recipe_obj = recipe_registry.get(recipe)
            if recipe_obj:
                tools = sorted(recipe_obj.get_all_tool_maker_names())
                output_lines = [f"  {recipe_obj.short_name}.{tool}" for tool in tools]
                return {
                    "success": True,
                    "exit_code": 0,
                    "stdout": "\n".join(output_lines),
                    "stderr": "",
                }
            return {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Recipe '{recipe}' not found",
            }

        elif tool_type:
            if tool_type not in tool_registry.name_to_tool:
                return {
                    "success": False,
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": f"Tool type '{tool_type}' not found",
                }

            recipes = recipe_registry.get_all()
            output_lines = []
            for r in sorted(recipes, key=lambda x: x.module_name):
                makers = r.get_makers_for_tool(tool_type)
                if makers:
                    for maker_name, _ in makers:
                        output_lines.append(f"  {r.short_name}.{maker_name}")

            if output_lines:
                return {
                    "success": True,
                    "exit_code": 0,
                    "stdout": "\n".join(output_lines),
                    "stderr": "",
                }
            return {
                "success": True,
                "exit_code": 0,
                "stdout": f"No recipes found supporting '{tool_type}'",
                "stderr": "",
            }

        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": "Either recipe or tool_type must be provided",
        }

    async def execute_help_command(self, tool_path: str) -> dict[str, Any]:
        """Get tool arguments using get_tool_arguments.

        Args:
            tool_path: Tool path (e.g., 'train arena')

        Returns:
            Dictionary with help output
        """
        try:
            result_json = await get_tool_arguments(tool_path)
            try:
                error_response = ErrorResponse.model_validate_json(result_json)
                if error_response.status == "error":
                    result = ToolExecutionResult(
                        success=False,
                        exit_code=1,
                        stdout="",
                        stderr=error_response.message,
                        command=f"./tools/run.py {tool_path} --help",
                        error="help_error",
                        summary=f"Error getting tool arguments: {error_response.message}",
                    )
                    return result.model_dump()
            except Exception:
                pass

            result = ToolExecutionResult(
                success=True,
                exit_code=0,
                stdout=result_json,
                stderr="",
                command=f"./tools/run.py {tool_path} --help",
                summary="Tool arguments retrieved successfully",
            )
            return result.model_dump()
        except Exception as e:
            logger.exception("Error executing help command")
            result = ToolExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                command=f"./tools/run.py {tool_path} --help",
                error="help_error",
                summary=f"Error executing help command: {str(e)}",
            )
            return result.model_dump()
