"""Guards for training pipelines - execution control decorators."""

from __future__ import annotations

import functools
import logging
import platform
from typing import Any, Callable, TypeVar

import torch

from metta.common.util.lock import broadcast_state
from metta.common.wandb.wandb_context import WandbContext

F = TypeVar("F", bound=Callable[..., Any])


def wandb_context(master_only: bool = True) -> Callable[[F], F]:
    """Guard that wraps a function in a WandB context.

    Args:
        master_only: Only create context on master process (rank 0)

    Returns:
        Decorated function that runs within WandB context
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(state, *args, **kwargs):
            # Handle both dict and object state
            if hasattr(state, "config"):
                # Object state with proper typing
                tool = state.config.tool
                torch_dist_cfg = getattr(state, "torch_dist_cfg", None)
            elif isinstance(state, dict):
                # Dict state (for backwards compatibility)
                tool = state.get("tool")
                torch_dist_cfg = state.get("torch_dist_cfg")
            else:
                # Other types (like Tool instances) - skip wandb
                return func(state, *args, **kwargs)

            # Skip if not master and master_only is True
            if master_only:
                if torch_dist_cfg and not torch_dist_cfg.is_master:
                    # Just pass through without wandb
                    return func(state, *args, **kwargs)

            # Create and enter WandB context
            if tool and tool.wandb:
                with WandbContext(tool.wandb, tool) as wandb_run:
                    if hasattr(state, "wandb_run"):
                        # State has wandb_run field (via mixin or similar)
                        old_wandb_run = state.wandb_run
                        state.wandb_run = wandb_run

                        # Update policy store if it exists
                        if hasattr(state, "policy_store") and state.policy_store:
                            state.policy_store.wandb_run = wandb_run

                        try:
                            result = func(state, *args, **kwargs)
                        finally:
                            # Restore original wandb_run value
                            state.wandb_run = old_wandb_run
                        return result
                    else:
                        # Dict state (backwards compatibility)
                        state_with_wandb = {**state, "wandb_run": wandb_run}
                        if "policy_store" in state:
                            state["policy_store"].wandb_run = wandb_run
                        return func(state_with_wandb, *args, **kwargs)
            else:
                # No wandb configured, just run the function
                return func(state, *args, **kwargs)

        return wrapper

    return decorator


def distributed_only() -> Callable[[F], F]:
    """Guard that only runs function if distributed training is initialized."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if torch.distributed.is_initialized():
                return func(*args, **kwargs)
            return args[0] if args else None  # Return state unchanged

        return wrapper

    return decorator


def master_process_only() -> Callable[[F], F]:
    """Guard that only runs function on the master process (rank 0).

    After the master process executes the function, the resulting state
    is broadcast to all other processes to prevent state drift.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # For methods, the first arg is self, second is state
            # For functions, the first arg is state
            # We need to handle both cases
            if len(args) >= 1:
                # Check if it's a method call (first arg is self)
                if hasattr(args[0].__class__, func.__name__):
                    # It's a method, state might be second arg
                    self_arg = args[0]
                    # For simplicity, always execute on master when it's a method
                    # The actual distributed check will be done inside the method if needed
                    is_master = True
                    if torch.distributed.is_initialized():
                        is_master = torch.distributed.get_rank() == 0
                else:
                    # It's a function, first arg is state
                    state = args[0]
                    # Handle both dict and object state
                    if hasattr(state, "torch_dist_cfg"):
                        # Object state (TrainingState)
                        torch_dist_cfg = state.torch_dist_cfg
                    elif isinstance(state, dict):
                        # Dict state (backwards compatibility)
                        torch_dist_cfg = state.get("torch_dist_cfg")
                    else:
                        # Other types of state - assume master
                        torch_dist_cfg = None

                    # Check if we're the master process
                    is_master = True
                    if torch_dist_cfg:
                        is_master = torch_dist_cfg.is_master
                    elif torch.distributed.is_initialized():
                        is_master = torch.distributed.get_rank() == 0
            else:
                # No args, just execute
                is_master = True
                if torch.distributed.is_initialized():
                    is_master = torch.distributed.get_rank() == 0

            if is_master:
                # Master process executes the function
                result = func(*args, **kwargs)
            else:
                # Non-master processes don't execute
                # For methods, return None; for functions with state, return state
                if len(args) >= 1 and not hasattr(args[0].__class__, func.__name__):
                    result = args[0]  # Return state unchanged
                else:
                    result = None

            # Broadcast the result from master to all workers if it's a state object
            if torch.distributed.is_initialized() and result is not None:
                # Only broadcast if it looks like a state object
                if hasattr(result, "__dict__") or isinstance(result, dict):
                    result = broadcast_state(result, src=0)

            return result

        return wrapper

    return decorator


def with_logging(log_prefix: str = "") -> Callable[[F], F]:
    """Guard that adds logging around a function execution.

    Args:
        log_prefix: Prefix for log messages

    Returns:
        Decorated function with logging
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)

            # Only log on master
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                stage_name = func.__name__.replace("_", " ").title()
                if log_prefix:
                    logger.info(f"{log_prefix}: Starting {stage_name}")
                else:
                    logger.info(f"Starting {stage_name}")

            result = func(*args, **kwargs)

            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                if log_prefix:
                    logger.info(f"{log_prefix}: Completed {stage_name}")
                else:
                    logger.info(f"Completed {stage_name}")

            return result

        return wrapper

    return decorator


def platform_specific(platform_name: str) -> Callable[[F], F]:
    """Guard that only runs function on specific platforms.

    Args:
        platform_name: Platform name (e.g., "Darwin", "Linux", "Windows")

    Returns:
        Decorated function that only runs on specified platform
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if platform.system() == platform_name:
                return func(*args, **kwargs)
            # Not the right platform, pass through first arg (state)
            return args[0] if args else None

        return wrapper

    return decorator
