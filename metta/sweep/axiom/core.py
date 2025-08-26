"""Core tAXIOM DSL components: Pipeline, Stage, and Context - MVP version."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from metta.sweep.axiom.types import infer_type

T = TypeVar("T")


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
    """Wrapper for a function that operates as a pipeline stage.

    Stages are the atomic units of computation in tAXIOM pipelines.
    Each stage:
    - Has a unique name
    - Wraps a function (method or standalone)
    - Has optional input/output type contracts (for documentation)
    - I/O stages have timeout and retry configuration
    
    Note: Checks and hooks are applied at membranes (via through()), not owned by stages
    """

    name: str
    func: Callable
    stage_type: str = "stage"  # "stage" or "io"
    input_type: type | None = None
    output_type: type | None = None
    hooks: list[Any] = field(default_factory=list)
    checks: list[Any] = field(default_factory=list)
    # I/O specific attributes
    timeout: float | None = None
    num_retries: int = 0

    def __post_init__(self):
        """Validate stage configuration."""
        if not callable(self.func):
            raise TypeError(f"Stage '{self.name}' func must be callable")

    def execute(self, ctx: Ctx, *args, **kwargs) -> Any:
        """Execute the stage function.
        
        Stage execution patterns:
        1. Methods bound to objects with self.cfg:
           - Called directly, can access self.cfg internally
           
        2. Lambdas and pure functions:
           - Receive previous stage's output as first argument
           - Config must be closed over in lambda or passed explicitly:
             lambda data: process(data, config)
           
        3. Context-aware functions (deprecated in MVP):
           - Would receive ctx directly (removed for simplicity)
        """
        # Get input from context if no args provided
        if not args and not kwargs:
            stage_input = ctx.get_stage_input(self.name)
            # Always pass the stage input, even if it's None (first stage needs it)
            args = (stage_input,)

        try:
            # Execute the function
            result = self.func(*args, **kwargs)
        except Exception as e:
            # Add context about which stage failed
            stage_type = "I/O operation" if self.stage_type == "io" else "Stage"
            error_msg = f"{stage_type} '{self.name}' failed: {str(e)}"
            
            # Create a new exception with the stage context
            # Use the same exception type if it's a standard one, otherwise use RuntimeError
            if isinstance(e, (TypeError, ValueError, KeyError, AttributeError, IndexError)):
                raise type(e)(error_msg) from e
            elif isinstance(e, IOError):
                # IOError from I/O operations already has context, just re-raise
                raise
            else:
                raise RuntimeError(error_msg) from e

        # Store output in context
        ctx.set_stage_output(self.name, result)
        ctx._current_stage = self.name

        return result


class Pipeline:
    """Pipeline for composing stages - MVP version with sequential execution only.

    Pipelines provide:
    - Method chaining API for building computation graphs
    - Sequential execution
    - Type documentation at membranes
    - Hook attachment points
    - Join points for swappable sub-pipelines
    
    Usage patterns for accessing configuration:
    
    1. Method-based (preferred for complex logic):
       ```python
       class MyExperiment:
           def __init__(self, cfg):
               self.cfg = cfg
           
           def process_data(self, data):
               # Can access self.cfg directly
               return data * self.cfg.multiplier
       
       pipeline.stage("process", self.process_data)
       ```
       
    2. Lambda with closed-over config:
       ```python
       cfg = config  # Close over config in lambda scope
       pipeline.stage("transform", lambda data: data * cfg.multiplier)
       ```
       
    3. External pure function with partial application:
       ```python
       from functools import partial
       pipeline.stage("validate", partial(external_validator, threshold=cfg.threshold))
       ```
       
    Note: The MVP removes context_aware decorators. All configuration access
    must be explicit through one of the patterns above.
    """

    def __init__(self, ctx: Ctx | None = None, name: str = ""):
        self.stages: list[Stage] = []
        self.stage_names: set[str] = set()
        self._built = False
        self._ctx = ctx
        self.name = name
        self._required_joins: dict[str, dict] = {}  # name -> {exit_checks, optional}
        self._provided_joins: dict[str, Pipeline] = {}  # name -> sub-pipeline
        
        # Exposure and override tracking
        self._exposed: set[str] = set()  # Components marked as overrideable
        self._overrides: dict[str, Callable | Pipeline] = {}  # Active overrides
        self._parent_path: str = ""  # For nested pipeline paths

    def stage(self, name: str, func: Callable, expose: bool = False, infer_types: bool = True) -> Pipeline:
        """Add a stage to the pipeline.

        Args:
            name: Unique name for the stage
            func: Function to execute (method or standalone)
            infer_types: Whether to attempt type inference from function annotations

        Returns:
            Self for method chaining
        """
        # Create stage
        stage_obj = Stage(name=name, func=func, stage_type="stage")

        # Attempt type inference if enabled
        if infer_types and func is not None:
            input_type, output_type = infer_type(func)
            
            if input_type is not None and stage_obj.input_type is None:
                stage_obj.input_type = input_type
            if output_type is not None and stage_obj.output_type is None:
                stage_obj.output_type = output_type

        self.stages.append(stage_obj)
        self.stage_names.add(name)
        
        # Mark as exposed if requested
        if expose:
            self._exposed.add(name)

        return self

    def io(self, name: str, func: Callable, expose: bool = False, timeout: float | None = None, num_retries: int = 0) -> Pipeline:
        """Add an I/O operation to the pipeline with error handling.

        I/O operations are wrapped with timeout and retry logic. They fail hard
        (propagate errors) after retries are exhausted, as they represent
        critical external interactions.
        
        IMPORTANT: Like all pipeline stages, I/O operations receive the output
        from the previous stage as their first argument. This enforces the
        node->node data flow contract.

        Args:
            name: Unique name for the I/O operation
            func: Function that performs external I/O (must accept previous stage output)
            timeout: Optional timeout in seconds for the operation
            num_retries: Number of retry attempts on failure (default: 0)

        Returns:
            Self for method chaining
        """
        import functools
        import threading
        import time
        
        def run_with_timeout(func, args, kwargs, timeout_seconds):
            """Run a function with a timeout using threading."""
            if timeout_seconds is None or timeout_seconds <= 0:
                return func(*args, **kwargs)
            
            result: list[Any] = [None]
            exception: list[Exception | None] = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                # Thread is still running, timeout occurred
                # Note: The thread will continue running in background
                raise TimeoutError(f"I/O operation '{name}' timed out after {timeout_seconds} seconds")
            
            if exception[0] is not None:
                raise exception[0]
            
            return result[0]
        
        @functools.wraps(func)
        def io_wrapper(*args, **kwargs):
            """Wrapper that adds timeout and retry logic to I/O operations.
            
            This wrapper preserves the node->node data flow contract - it accepts
            the previous stage's output and passes it through to the I/O function.
            """
            last_exception = None
            
            for attempt in range(num_retries + 1):
                try:
                    # Run with timeout if specified
                    # Pass through all arguments - preserves data flow
                    return run_with_timeout(func, args, kwargs, timeout)
                        
                except Exception as e:
                    last_exception = e
                    
                    # If this was the last attempt, propagate the error
                    if attempt >= num_retries:
                        # Fail hard - I/O errors should propagate
                        error_msg = f"I/O operation '{name}' failed after {num_retries + 1} attempts"
                        if timeout:
                            error_msg += f" (timeout: {timeout}s)"
                        error_msg += f": {str(e)}"
                        
                        # Re-raise with additional context
                        raise IOError(error_msg) from e
                    
                    # Log retry attempt
                    print(f"I/O operation '{name}' failed (attempt {attempt + 1}/{num_retries + 1}): {e}")
                    
                    # Wait before retry (exponential backoff)
                    wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                    time.sleep(wait_time)
            
            # This should never be reached, but included for completeness
            if last_exception:
                raise last_exception
        
        # Create I/O stage with wrapped function
        stage_obj = Stage(name=name, func=io_wrapper, stage_type="io")
        
        # Store I/O configuration as attributes for inspection
        stage_obj.timeout = timeout
        stage_obj.num_retries = num_retries
        
        self.stages.append(stage_obj)
        self.stage_names.add(name)
        
        # Mark as exposed if requested
        if expose:
            self._exposed.add(name)

        return self

    def through(
        self, 
        output_type: type | None = None, 
        input_type: type | None = None, 
        checks: list | dict | None = None,
        hooks: list[Any] | None = None, 
        **kwargs
    ) -> Pipeline:
        """Create a membrane (no-op stage) with type contracts, checks, and hooks.

        This creates a pass-through stage that serves as the membrane where:
        - Type contracts are documented
        - Checks validate data integrity
        - Hooks are attached for observation

        Args:
            output_type: Expected output type for documentation
            input_type: Expected input type (optional)
            checks: List of checks or dict of named checks
            hooks: List of hooks to attach
            **kwargs: Additional configuration

        Returns:
            Self for method chaining
        """
        if not self.stages:
            raise ValueError("No stage to apply 'through' to")

        last_stage = self.stages[-1]
        
        # Set type contracts on the previous stage
        if output_type is not None:
            last_stage.output_type = output_type

        if input_type is not None:
            last_stage.input_type = input_type

        # Create a no-op membrane stage with checks and hooks
        membrane_name = f"_membrane_{last_stage.name}"
        membrane_stage = Stage(
            name=membrane_name,
            func=lambda x: x,  # Pass-through function
            stage_type="membrane",
            input_type=output_type,  # Input matches previous stage's output
            output_type=output_type,  # Output is the same
            hooks=hooks if hooks else [],
            checks=checks if isinstance(checks, list) else []
        )
        
        # Handle dict format for checks
        if checks and isinstance(checks, dict):
            membrane_stage.checks = []
            for name, check in checks.items():
                check.name = name
                membrane_stage.checks.append(check)
        elif checks:
            membrane_stage.checks = list(checks)
        
        self.stages.append(membrane_stage)
        self.stage_names.add(membrane_name)

        return self

    def T(self, output_type: type) -> Pipeline:
        """Shorthand for through(output_type)."""
        return self.through(output_type)

    def hook(self, func: Callable) -> Pipeline:
        """Add a standalone hook to the pipeline.

        Note: Prefer attaching hooks at membranes via through()

        Args:
            func: Hook function to execute

        Returns:
            Self for method chaining
        """
        if not self.stages:
            raise ValueError("No stage to attach hook to")
        
        # Attach to the last stage
        self.stages[-1].hooks.append(func)
        return self

    def join(self, name: str, sub: Pipeline | Callable | None = None, propagate_global_checks: bool = False, expose: bool = True) -> Pipeline:
        """Add a join point for a sub-pipeline.

        Args:
            name: Name of the join point
            sub: Sub-pipeline or callable to execute (can be None if using require_join pattern)
            propagate_global_checks: Whether to propagate global checks into the sub-pipeline
            expose: Whether this join can be overridden (default: True for joins)

        Returns:
            Self for method chaining
        """
        # If callable but not Pipeline, wrap it
        if callable(sub) and not isinstance(sub, Pipeline):
            wrapped = Pipeline(name=f"{name}_wrapped")
            wrapped.stage(name, sub)
            sub = wrapped
        
        # Mark as exposed if requested (joins default to exposed)
        if expose:
            self._exposed.add(name)
        
        # Store the sub-pipeline
        if sub is not None:
            self._provided_joins[name] = sub
        
        if sub is None:
            # Check if this was a required join
            if name not in self._required_joins:
                raise ValueError(f"Join '{name}' has no sub-pipeline and was not required")
            # Use provided implementation if available
            if name in self._provided_joins:
                sub = self._provided_joins[name]
            else:
                # Optional join without implementation - use empty pipeline
                if self._required_joins[name].get("optional", False):
                    sub = Pipeline()
                else:
                    raise ValueError(f"Required join '{name}' has no implementation")
        
        # Create a stage that executes the sub-pipeline
        def execute_sub(data):
            sub_ctx = Ctx()
            sub_ctx.set_stage_input("_join_input", data)
            result = sub.run(sub_ctx)
            return result
        
        stage_obj = Stage(name=f"join:{name}", func=execute_sub, stage_type="stage")
        self.stages.append(stage_obj)
        self.stage_names.add(f"join:{name}")
        
        return self

    def require_join(self, name: str, exit_checks: list | None = None, optional: bool = False) -> Pipeline:
        """Declare a required join point that must be provided.

        Args:
            name: Name of the join point
            exit_checks: Checks to run on the output of the provided implementation
            optional: Whether this join is optional

        Returns:
            Self for method chaining
        """
        self._required_joins[name] = {
            "exit_checks": exit_checks or [],
            "optional": optional
        }
        
        # Add placeholder join
        return self.join(name, None)

    def provide_join(self, name: str, sub: Pipeline) -> Pipeline:
        """Provide an implementation for a required join.

        Args:
            name: Name of the join point to fill
            sub: Sub-pipeline implementation

        Returns:
            Self for method chaining
        """
        if name not in self._required_joins:
            raise ValueError(f"Join '{name}' was not required")
        
        self._provided_joins[name] = sub
        return self
    
    def compose(self, sub: Pipeline) -> Pipeline:
        """Compose another pipeline into this one.
        
        This directly adds all stages from the sub-pipeline to this pipeline,
        preserving the data flow between them.
        
        Args:
            sub: Sub-pipeline to compose
        
        Returns:
            Self for method chaining
        """
        for stage in sub.stages:
            # Add the stage directly
            self.stages.append(stage)
            self.stage_names.add(stage.name)
        
        return self

    def logf(self, format_string: str) -> Pipeline:
        """Add inline logging without lambdas.

        Args:
            format_string: Format string with {payload.field} or {ctx.metadata.field} placeholders

        Returns:
            Self for method chaining
        """
        import logging
        import re
        
        logger = logging.getLogger(__name__)
        
        def log_hook(result, ctx):
            # Parse format string for patterns
            formatted = format_string
            
            # Replace {payload.field} with actual values from result
            payload_pattern = r'\{payload\.(\w+)\}'
            payload_matches = re.findall(payload_pattern, format_string)
            
            for field_name in payload_matches:
                if isinstance(result, dict):
                    value = result.get(field_name, "N/A")
                else:
                    value = getattr(result, field_name, "N/A")
                
                # Format floats nicely
                if isinstance(value, float):
                    formatted = formatted.replace(f"{{payload.{field_name}}}", f"{value:.4f}")
                else:
                    formatted = formatted.replace(f"{{payload.{field_name}}}", str(value))
            
            # Replace {ctx.metadata.field} with actual values from context
            ctx_pattern = r'\{ctx\.metadata\.(\w+)\}'
            ctx_matches = re.findall(ctx_pattern, format_string)
            
            for field_name in ctx_matches:
                value = ctx.metadata.get(field_name, "N/A")
                formatted = formatted.replace(f"{{ctx.metadata.{field_name}}}", str(value))
            
            # Use logging module instead of print
            logger.info(formatted)
        
        return self.hook(log_hook)

    def build(self) -> Pipeline:
        """Build the pipeline graph (currently just validation)."""
        if self._built:
            return self

        # Validate stage configuration
        for stage_obj in self.stages:
            if stage_obj.func is None:
                raise ValueError(f"Stage '{stage_obj.name}' has no function")

        self._built = True
        return self

    def run(self, ctx: Ctx | None = None) -> Any:
        """Execute the pipeline sequentially.

        Args:
            ctx: Context object (created if not provided)

        Returns:
            Output from the last stage
        """
        # Build if not already built
        if not self._built:
            self.build()

        # Use provided context or create new one
        if ctx is None:
            ctx = self._ctx if self._ctx is not None else Ctx()
        self._ctx = ctx

        # Execute stages in sequence
        last_result = None
        for i, stage_obj in enumerate(self.stages):
            # Set input from previous stage output
            if i > 0 and last_result is not None:
                ctx.set_stage_input(stage_obj.name, last_result)
            elif i == 0:
                # First stage gets None as input if nothing else is provided
                ctx.set_stage_input(stage_obj.name, None)

            # Check for override
            stage_name = stage_obj.name
            
            # For join stages, extract the join name
            if stage_name.startswith("join:"):
                join_name = stage_name[5:]  # Remove "join:" prefix
                if join_name in self._exposed and join_name in self._overrides:
                    # Execute override instead
                    override_func = self._overrides[join_name]
                    stage_input = ctx.get_stage_input(stage_obj.name)
                    if isinstance(override_func, Pipeline):
                        sub_ctx = Ctx()
                        sub_ctx.set_stage_input("_join_input", stage_input)
                        result = override_func.run(sub_ctx)
                    else:
                        result = override_func(stage_input)
                    ctx.set_stage_output(stage_obj.name, result)
                    ctx._current_stage = stage_obj.name
                else:
                    # Execute normally (join will handle its own overrides internally)
                    result = stage_obj.execute(ctx)
            elif stage_name in self._exposed and stage_name in self._overrides:
                # Execute override for regular stage
                override_func = self._overrides[stage_name]
                stage_input = ctx.get_stage_input(stage_obj.name)
                result = override_func(stage_input)
                ctx.set_stage_output(stage_obj.name, result)
                ctx._current_stage = stage_obj.name
            else:
                # Execute stage normally
                result = stage_obj.execute(ctx)

            # Run checks if present
            if hasattr(stage_obj, 'checks'):
                from metta.sweep.axiom.checks import CheckLevel
                
                for check in stage_obj.checks:
                    passed, error_msg = check.check(result)
                    if not passed:
                        if check.level == CheckLevel.FAIL:
                            raise ValueError(f"Check failed at stage '{stage_obj.name}': {error_msg}")
                        else:
                            # Just warn for now (could use logging)
                            print(f"Warning at stage '{stage_obj.name}': {error_msg}")

            # Run hooks (non-mutating, observational only)
            for hook in stage_obj.hooks:
                if callable(hook):
                    try:
                        # Hooks receive result and context
                        hook(result, ctx)
                    except Exception as e:
                        # Hooks fail silently - log error and continue
                        hook_name = getattr(hook, '__name__', str(hook))
                        print(f"Warning: Hook '{hook_name}' at stage '{stage_obj.name}' failed: {e}")
                        # Continue with pipeline execution - hooks are observational only

            last_result = result

        return last_result
    
    def override(self, path: str, replacement: Callable | Pipeline) -> Pipeline:
        """Override an exposed component at any depth.
        
        Only components marked with expose=True can be overridden.
        
        Args:
            path: Dot-separated path to the component (e.g., "train.compute_advantage")
            replacement: Callable or pipeline to use instead
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If trying to override a non-exposed component
        """
        parts = path.split(".", 1)
        
        if len(parts) == 1:
            # Direct override at this level
            component_name = parts[0]
            
            # Check if it's exposed
            if component_name not in self._exposed:
                available = ", ".join(sorted(self._exposed)) if self._exposed else "none"
                raise ValueError(
                    f"Cannot override '{component_name}' - not exposed. "
                    f"Available: {available}"
                )
            
            # Store the override
            self._overrides[component_name] = replacement
            
        else:
            # Nested override
            join_name, nested_path = parts
            
            # Check if the join exists and is exposed
            if join_name not in self._provided_joins:
                raise ValueError(f"No join named '{join_name}' in pipeline")
            
            if join_name not in self._exposed:
                raise ValueError(
                    f"Cannot override into '{join_name}' - join not exposed. "
                    f"Add expose=True to the join() call."
                )
            
            # Propagate to the nested pipeline
            nested_pipeline = self._provided_joins[join_name]
            if isinstance(nested_pipeline, Pipeline):
                nested_pipeline.override(nested_path, replacement)
            else:
                raise ValueError(
                    f"Cannot override nested path '{nested_path}' in non-pipeline join '{join_name}'"
                )
        
        return self
    
    def list_exposed(self, prefix: str = "") -> list[str]:
        """List all exposed components that can be overridden.
        
        Returns:
            List of dot-separated paths to exposed components
        """
        exposed = []
        
        # Add directly exposed components
        for name in self._exposed:
            full_name = f"{prefix}.{name}" if prefix else name
            exposed.append(full_name)
            
            # If it's a join with a pipeline, recurse
            if name in self._provided_joins:
                sub_pipeline = self._provided_joins[name]
                if isinstance(sub_pipeline, Pipeline):
                    nested_exposed = sub_pipeline.list_exposed(full_name)
                    exposed.extend(nested_exposed)
        
        return exposed
    
    def get_overrides(self) -> dict[str, Any]:
        """Get all active overrides at all levels.
        
        Returns:
            Dictionary of override paths and their replacements
        """
        overrides = dict(self._overrides)
        
        # Add nested overrides
        for join_name, sub_pipeline in self._provided_joins.items():
            if isinstance(sub_pipeline, Pipeline):
                nested_overrides = sub_pipeline.get_overrides()
                for nested_path, replacement in nested_overrides.items():
                    full_path = f"{join_name}.{nested_path}"
                    overrides[full_path] = replacement
        
        return overrides