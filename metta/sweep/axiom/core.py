"""Core tAXIOM DSL components: Pipeline, Stage, and Context."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from metta.sweep.axiom.types import infer_type, validate_type

T = TypeVar("T")


def context_aware(func: Callable) -> Callable:
    """Decorator to mark a function as needing the full context object.

    Usage:
        @context_aware
        def my_stage(ctx: Ctx) -> dict:
            # Can access full context
            return {"budget": ctx.metadata.get("budget")}

    Or inline:
        pipeline.stage("my_stage", context_aware(lambda ctx: {...}))
    """
    func.__context_aware__ = True
    return func


class Ctx:
    """Context object that flows through pipeline stages.

    The Ctx object manages data flow between stages, maintaining:
    - Input/output for each stage
    - Metadata (seeds, hashes, etc.)
    - Artifacts (checkpoints, logs)
    """

    def __init__(self):
        self.stages: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, Any] = {}
        self.artifacts: dict[str, Any] = {}
        self._current_stage: str | None = None

    def set_stage_input(self, stage_name: str, input_data: Any) -> None:
        """Set input data for a stage."""
        if stage_name not in self.stages:
            self.stages[stage_name] = {}
        self.stages[stage_name]["in"] = input_data

    def set_stage_output(self, stage_name: str, output_data: Any) -> None:
        """Set output data from a stage."""
        if stage_name not in self.stages:
            self.stages[stage_name] = {}
        self.stages[stage_name]["out"] = output_data

    def get_stage_input(self, stage_name: str) -> Any:
        """Get input data for a stage."""
        return self.stages.get(stage_name, {}).get("in")

    def get_stage_output(self, stage_name: str) -> Any:
        """Get output data from a stage."""
        return self.stages.get(stage_name, {}).get("out")

    def get_last_output(self) -> Any:
        """Get output from the most recently executed stage."""
        if self._current_stage:
            return self.get_stage_output(self._current_stage)
        return None


@dataclass
class Stage:
    """Wrapper for a pure function that operates as a pipeline stage.

    Stages are the atomic units of computation in tAXIOM pipelines.
    Each stage:
    - Has a unique name
    - Wraps a pure function
    - Has optional input/output type contracts
    - Can have hooks attached at its membrane
    """

    name: str
    func: Callable
    input_type: type | None = None
    output_type: type | None = None
    hooks: list[Any] = field(default_factory=list)

    def __post_init__(self):
        """Validate stage configuration."""
        if not callable(self.func):
            raise TypeError(f"Stage '{self.name}' func must be callable")

    def execute(self, ctx: Ctx, *args, **kwargs) -> Any:
        """Execute the stage function with context."""
        # Check if function is marked as context-aware
        is_context_aware = getattr(self.func, "__context_aware__", False)

        # Get input from context if no args provided
        if not args and not kwargs and not is_context_aware:
            stage_input = ctx.get_stage_input(self.name)
            if stage_input is not None:
                # Always pass as single argument - let the function handle unpacking
                args = (stage_input,)

        # Execute the function
        if is_context_aware:
            # Pass context as first argument
            result = self.func(ctx, *args, **kwargs)
        else:
            # Normal execution with just the input data
            result = self.func(*args, **kwargs)

        # Store output in context
        ctx.set_stage_output(self.name, result)
        ctx._current_stage = self.name

        return result

    def validate_input(self, data: Any) -> None:
        """Validate input data against type contract."""
        if self.input_type is None:
            return

        try:
            validate_type(data, self.input_type)
        except TypeError as e:
            raise TypeError(f"Stage '{self.name}' input validation failed: {e}") from e

    def validate_output(self, data: Any) -> None:
        """Validate output data against type contract."""
        if self.output_type is None:
            return

        try:
            validate_type(data, self.output_type)
        except TypeError as e:
            raise TypeError(f"Stage '{self.name}' output validation failed: {e}") from e


class Pipeline:
    """Pipeline for composing stages with control flow.

    Pipelines provide:
    - Method chaining API for building computation graphs
    - Lazy evaluation (build first, execute later)
    - Type contract enforcement at membranes
    - Hook attachment points
    - Control flow primitives
    """

    def __init__(self):
        self.stages: list[Stage] = []
        self.stage_names: set[str] = set()
        self._built = False
        self._ctx: Ctx | None = None

    def stage(self, name: str, func: Callable, infer_types: bool = True) -> Pipeline:
        """Add a stage to the pipeline.

        Args:
            name: Unique name for the stage
            func: Pure function to execute (optional if name is callable)
            infer_types: Whether to attempt type inference from function annotations

        Returns:
            Self for method chaining
        """

        # Check for duplicate names
        if name in self.stage_names:
            warnings.warn(f"Duplicate stage name '{name}' - this may cause issues", stacklevel=2)

        # Create stage
        stage_obj = Stage(name=name, func=func)

        # Attempt type inference if enabled
        if infer_types and func is not None:
            input_type, output_type = infer_type(func)

            # Don't infer input type for context-aware functions
            # (they get Ctx, but that's not the stage input type)
            is_context_aware = getattr(func, "__context_aware__", False)

            if input_type is not None and stage_obj.input_type is None and not is_context_aware:
                stage_obj.input_type = input_type
            if output_type is not None and stage_obj.output_type is None:
                stage_obj.output_type = output_type

        self.stages.append(stage_obj)
        self.stage_names.add(name)

        return self

    def through(
        self, output_type: type, input_type: type | None = None, hooks: list[Any] | None = None, **hook_kwargs
    ) -> Pipeline:
        """Define type contract and hooks for the previous stage.

        Args:
            output_type: Expected output type for validation
            input_type: Expected input type (optional)
            hooks: List of hooks to attach
            **hook_kwargs: Additional hook configuration

        Returns:
            Self for method chaining
        """
        if not self.stages:
            raise ValueError("No stage to apply 'through' to")

        last_stage = self.stages[-1]
        last_stage.output_type = output_type

        if input_type is not None:
            last_stage.input_type = input_type

        if hooks:
            last_stage.hooks.extend(hooks)

        return self

    def T(self, output_type: type) -> Pipeline:
        """Shorthand for through(output_type)."""
        return self.through(output_type)

    def build(self) -> Pipeline:
        """Build the pipeline graph (lazy evaluation)."""
        if self._built:
            return self

        # Validate stage configuration
        for stage_obj in self.stages:
            if stage_obj.func is None:
                raise ValueError(f"Stage '{stage_obj.name}' has no function")

        self._built = True
        return self

    def run(self, ctx: Ctx | None = None) -> Any:
        """Execute the pipeline.

        Args:
            ctx: Context object (created if not provided)

        Returns:
            Output from the last stage
        """
        # Build if not already built
        if not self._built:
            self.build()

        # Create or use provided context
        if ctx is None:
            ctx = Ctx()
        self._ctx = ctx

        # Execute stages in sequence
        last_result = None
        for i, stage_obj in enumerate(self.stages):
            # Set input from previous stage output
            if i > 0 and last_result is not None:
                ctx.set_stage_input(stage_obj.name, last_result)

            # Get input for validation
            stage_input = ctx.get_stage_input(stage_obj.name)

            # Validate input
            if stage_input is not None:
                stage_obj.validate_input(stage_input)

            # Execute stage
            result = stage_obj.execute(ctx)

            # Validate output
            stage_obj.validate_output(result)

            # Run hooks (non-mutating)
            for hook in stage_obj.hooks:
                if hasattr(hook, "run"):
                    hook.run(ctx, stage_obj.name, result)

            last_result = result

        return last_result
