<<<<<<< Updated upstream
"""Executor for running tools programmatically using existing utilities."""

import copy
import inspect
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from metta.common.tool import Tool
from metta.common.tool.recipe_registry import recipe_registry
from metta.common.tool.run_tool import (
    classify_remaining_args,
    deep_merge,
    get_function_params,
    get_tool_fields,
    nestify,
    parse_cli_args,
    type_parse,
)
from metta.common.tool.tool_path import parse_two_token_syntax, resolve_and_load_tool_maker
from metta.common.tool.tool_registry import tool_registry
from metta.rl.system_config import seed_everything

from .tools.run_tool import get_tool_arguments

=======
"""Executor for running run.py commands via subprocess."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

>>>>>>> Stashed changes
logger = logging.getLogger(__name__)


class RunToolExecutor:
<<<<<<< Updated upstream
    """Executes tools programmatically using existing run_tool utilities.

    Uses the same logic as run_tool.main() but executes directly without subprocesses.
    """
=======
    """Executes run.py commands and returns structured results."""
>>>>>>> Stashed changes

    def __init__(self, run_script_path: Path, repo_root: Path, timeout: int = 3600):
        """Initialize the executor.

        Args:
<<<<<<< Updated upstream
            run_script_path: Path to the run.py script (kept for compatibility, not used)
            repo_root: Root of the Metta repository
            timeout: Default timeout in seconds (for future async support)
=======
            run_script_path: Path to the run.py script
            repo_root: Root of the Metta repository
            timeout: Default timeout in seconds for command execution
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        """Execute a tool programmatically.

        Replicates run_tool.main() logic but executes directly without subprocess.
=======
        """Execute a run.py command.
>>>>>>> Stashed changes

        Args:
            tool_path: Tool path (e.g., 'train arena', 'arena.train')
            arguments: Dictionary of key=value arguments
            dry_run: If True, validate without executing
            verbose: If True, show verbose output
            timeout: Override default timeout

        Returns:
<<<<<<< Updated upstream
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

            # Normalize tool_path to handle two-token syntax like "train arena"
            tool_path_parts = tool_path.strip().split(None, 1)
            if len(tool_path_parts) == 2:
                # tool_path is already in two-token format (e.g., "train arena")
                first_token, second_token_from_path = tool_path_parts
                if "=" not in second_token_from_path and not second_token_from_path.startswith("-"):
                    # This is two-token syntax, resolve it
                    resolved_tool_path, _ = parse_two_token_syntax(first_token, second_token_from_path)
                    tool_path = resolved_tool_path  # Use resolved path for further processing

            cli_args_list: list[str] = []
            if arguments:
                for key, value in arguments.items():
                    if value is None:
                        continue
                    # Convert value to string (matching run_tool.py behavior)
                    if isinstance(value, bool):
                        value_str = str(value).lower()
                    elif isinstance(value, (list, dict)):
                        value_str = json.dumps(value)
                    else:
                        value_str = str(value)
                    cli_args_list.append(f"{key}={value_str}")

            # Parse tool path, handling two-token syntax
            second_token = None
            if cli_args_list and "=" not in cli_args_list[0] and not cli_args_list[0].startswith("-"):
                second_token = cli_args_list[0]

            resolved_tool_path, args_consumed = parse_two_token_syntax(tool_path, second_token)

            # Skip consumed args
            remaining_cli_args = cli_args_list[args_consumed:]

            # Build equivalent CLI command
            cmd_parts = ["./tools/run.py", resolved_tool_path]
            if remaining_cli_args:
                cmd_parts.extend(remaining_cli_args)
            equivalent_command = " ".join(cmd_parts)

            # Parse CLI arguments using existing utility
            try:
                cli_args = parse_cli_args(remaining_cli_args)
            except ValueError as e:
                return {
                    "success": False,
                    "exit_code": 2,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": f"Error parsing arguments: {e}",
                    "error": "argument_parse_error",
                    "command": equivalent_command,
                }

            # Build nested payload using existing utility
            nested_cli = nestify(cli_args)

            tool_maker = resolve_and_load_tool_maker(resolved_tool_path)
            if tool_maker is None:
                return {
                    "success": False,
                    "exit_code": 1,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": f"Could not find tool '{tool_path}'",
                    "error": "tool_not_found",
                    "command": equivalent_command,
                }

            if verbose:
                logger.info(f"Loading tool: {tool_maker.__module__}.{tool_maker.__name__}")

            func_args_for_invoke: dict[str, str] = {}
            try:
                if inspect.isclass(tool_maker) and issubclass(tool_maker, Tool):
                    if verbose and nested_cli:
                        logger.info(f"Creating {tool_maker.__name__} from nested CLI payload")
                        for k in sorted(nested_cli.keys()):
                            logger.info(f"  {k} = {nested_cli[k]}")
                    tool_cfg = tool_maker.model_validate(nested_cli)
                    remaining_args = {}
                else:
                    sig = inspect.signature(tool_maker)
                    func_kwargs: dict[str, Any] = {}
                    consumed_keys: set[str] = set()

                    if verbose and (cli_args or nested_cli):
                        func_name = getattr(tool_maker, "__name__", str(tool_maker))
                        logger.info(f"Creating {func_name}:")
                    for name, p in sig.parameters.items():
                        if name in nested_cli:
                            provided = nested_cli[name]
                            base: Any | None = None
                            if p.default is not inspect._empty:
                                default_val = p.default
                                if isinstance(default_val, dict) and isinstance(provided, dict):
                                    base = copy.deepcopy(default_val)
                                    deep_merge(base, provided)
                                elif isinstance(default_val, BaseModel) and isinstance(provided, dict):
                                    base = default_val.model_copy(update=provided, deep=True)

                            data = base if base is not None else provided
                            ann = p.annotation
                            try:
                                if inspect.isclass(ann) and issubclass(ann, BaseModel):
                                    val = ann.model_validate(data)
                                else:
                                    val = type_parse(data, ann)
                            except Exception:
                                val = data

                            func_kwargs[name] = val

                            if name in cli_args:
                                consumed_keys.add(name)
                            for k in cli_args.keys():
                                if k.startswith(name + "."):
                                    consumed_keys.add(k)

                            if verbose:
                                logger.info(f"  {name}={val!r}")
                            continue
                        if name in cli_args:
                            val = type_parse(cli_args[name], p.annotation)
                            func_kwargs[name] = val
                            consumed_keys.add(name)
                            if verbose:
                                logger.info(f"  {name}={val!r}")
                    tool_cfg = tool_maker(**func_kwargs)
                    remaining_args = {k: v for k, v in cli_args.items() if k not in consumed_keys}
                    func_args_for_invoke = {k: str(v) for k, v in func_kwargs.items()}

            except TypeError as e:
                msg = str(e)
                hint = ""
                if ("missing" in msg and "positional argument" in msg) and (" self" in msg or " cls" in msg):
                    hint = (
                        "\nHint: It looks like an unbound method was passed. "
                        "Pass the Tool subclass itself or a factory function that doesn't require 'self'/'cls'."
                    )
                return {
                    "success": False,
                    "exit_code": 1,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": f"Error creating tool configuration: {e}{hint}",
                    "error": "tool_construction_error",
                    "command": equivalent_command,
                }
            except Exception as e:
                logger.exception("Error creating tool configuration")
                return {
                    "success": False,
                    "exit_code": 1,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": f"Error creating tool configuration: {e}",
                    "error": "tool_construction_error",
                    "command": equivalent_command,
                }

            if not isinstance(tool_cfg, Tool):
                return {
                    "success": False,
                    "exit_code": 1,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": f"{tool_path} must return a Tool instance, got {type(tool_cfg)}",
                    "error": "invalid_tool_type",
                    "command": equivalent_command,
                }

            # Handle overrides and unknown arguments
            tool_fields = get_tool_fields(type(tool_cfg))
            override_args, unknown_args = classify_remaining_args(remaining_args, tool_fields)

            if unknown_args:
                error_msg = f"Unknown arguments: {', '.join(unknown_args)}"
                if not (inspect.isclass(tool_maker) and issubclass(tool_maker, Tool)):
                    error_msg += f"\nAvailable function parameters: {', '.join(get_function_params(tool_maker))}"
                error_msg += f"\nAvailable tool fields for overrides: {', '.join(sorted(tool_fields))}"
                return {
                    "success": False,
                    "exit_code": 2,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": error_msg,
                    "error": "unknown_arguments",
                    "command": equivalent_command,
                }

            if override_args:
                if verbose:
                    logger.info("Applying overrides:")
                    for key, value in override_args.items():
                        logger.info(f"  {key}={value}")
                for key, value in override_args.items():
                    try:
                        tool_cfg = tool_cfg.override(key, value)
                    except Exception as e:
                        return {
                            "success": False,
                            "exit_code": 1,
                            "stdout": stdout_capture.getvalue(),
                            "stderr": f"Error applying override {key}={value}: {e}",
                            "error": "override_error",
                            "command": equivalent_command,
                        }

            # Dry run check
            if dry_run:
                if verbose:
                    logger.info("✅ Configuration validation successful")
                    logger.info(f"Tool type: {type(tool_cfg).__name__}")
                    logger.info(f"Module: {tool_maker.__module__}.{tool_maker.__name__}")
                return {
                    "success": True,
                    "exit_code": 0,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                    "command": equivalent_command,
                }

            # Seed & Run
            if hasattr(tool_cfg, "system"):
                seed_everything(tool_cfg.system)

            if verbose:
                logger.info("Running tool...")

            try:
                result = tool_cfg.invoke(func_args_for_invoke)
                exit_code = result if result is not None else 0
            except KeyboardInterrupt:
                return {
                    "success": False,
                    "exit_code": 130,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": "Interrupted by Ctrl-C",
                    "error": "interrupted",
                    "command": equivalent_command,
                }
            except Exception as e:
                logger.exception("Tool invocation failed")
                return {
                    "success": False,
                    "exit_code": 1,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": f"Tool invocation failed: {e}",
                    "error": "invocation_error",
                    "command": equivalent_command,
                }
=======
            Dictionary with:
            - success: bool
            - exit_code: int
            - stdout: str
            - stderr: str
            - command: str (the actual command executed)
        """
        # Build command
        cmd = ["uv", "run", str(self.run_script_path), tool_path]

        # Add flags
        if dry_run:
            cmd.append("--dry-run")
        if verbose:
            cmd.append("--verbose")

        # Add arguments in key=value format
        if arguments:
            for key, value in arguments.items():
                if value is None:
                    continue
                # Convert value to string, handling special cases
                if isinstance(value, bool):
                    value_str = str(value).lower()
                elif isinstance(value, (list, dict)):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                cmd.append(f"{key}={value_str}")

        command_str = " ".join(cmd)
        logger.info(f"Executing: {command_str}")

        try:
            # Run subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            timeout_seconds = timeout or self.timeout
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout_seconds} seconds",
                    "command": command_str,
                    "error": "timeout",
                }

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            exit_code = process.returncode
>>>>>>> Stashed changes

            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
<<<<<<< Updated upstream
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "command": equivalent_command,  # Equivalent CLI command
            }

        finally:
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
=======
                "stdout": stdout_str,
                "stderr": stderr_str,
                "command": command_str,
            }

        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "command": command_str,
                "error": "execution_error",
            }
>>>>>>> Stashed changes

    async def execute_list_command(
        self, recipe: Optional[str] = None, tool_type: Optional[str] = None
    ) -> dict[str, Any]:
<<<<<<< Updated upstream
        """Execute a --list command to discover tools using recipe_registry.
=======
        """Execute a --list command to discover tools.
>>>>>>> Stashed changes

        Args:
            recipe: Recipe name to list tools for (e.g., 'arena')
            tool_type: Tool type to find recipes for (e.g., 'train')

        Returns:
            Dictionary with discovery results
        """
        if recipe:
<<<<<<< Updated upstream
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
=======
            cmd = ["uv", "run", str(self.run_script_path), recipe, "--list"]
        elif tool_type:
            cmd = ["uv", "run", str(self.run_script_path), tool_type, "--list"]
        else:
            return {
                "success": False,
                "error": "Either recipe or tool_type must be provided",
            }

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            return {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "command": " ".join(cmd),
            }

        except Exception as e:
            logger.error(f"Error executing list command: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def execute_help_command(self, tool_path: str) -> dict[str, Any]:
        """Execute a --help command to get tool arguments.
>>>>>>> Stashed changes

        Args:
            tool_path: Tool path (e.g., 'train arena')

        Returns:
            Dictionary with help output
        """
<<<<<<< Updated upstream
        try:
            result_json = await get_tool_arguments(tool_path)
            result = json.loads(result_json)
            if "status" in result and result["status"] == "error":
                return {
                    "success": False,
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": result.get("message", "Unknown error"),
                }
            return {
                "success": True,
                "exit_code": 0,
                "stdout": result_json,
                "stderr": "",
            }
        except Exception as e:
            logger.exception("Error executing help command")
            return {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": str(e),
            }
=======
        cmd = ["uv", "run", str(self.run_script_path), tool_path, "--help"]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            return {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "command": " ".join(cmd),
            }

        except Exception as e:
            logger.error(f"Error executing help command: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

>>>>>>> Stashed changes
