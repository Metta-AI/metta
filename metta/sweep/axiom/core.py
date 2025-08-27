"""Core tAXIOM DSL components: Stateful Pipeline with clean State/Context separation.

Key design principles:
1. State (Pydantic): Strongly-typed, mutable experiment data
2. Context (TypedDict): Lightweight, read-only execution metadata
3. Stages receive both: (state: State, ctx: Context) -> None
4. Deep overrides preserved via pipeline_path in Context
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, Field

from metta.sweep.axiom.types import infer_type

T = TypeVar("T")


class PipelineState(BaseModel):
    """Base class for pipeline state - subclass for specific pipelines.
    
    This uses Pydantic for type safety and validation while allowing flexibility.
    Subclass this to define your pipeline's specific state structure.
    """
    # Common fields all pipelines have
    metadata: dict[str, Any] = Field(default_factory=dict, description="User metadata")
    artifacts: dict[str, Any] = Field(default_factory=dict, description="Stored artifacts")
    outputs: dict[str, Any] = Field(default_factory=dict, description="Stage outputs for compatibility")
    
    class Config:
        extra = "allow"  # Allow dynamic fields when needed
        arbitrary_types_allowed = True  # Allow Any types like models


class Context(BaseModel):
    """Read-only execution context - stages should NOT modify this.
    
    While this is a Pydantic model (for consistency and validation),
    it's conceptually read-only - stages should treat it as immutable.
    """
    # Execution tracking
    current_stage: str = ""
    stage_history: list[str] = Field(default_factory=list)
    
    # Pipeline path for deep overrides (e.g., "parent.child.grandchild")
    pipeline_path: str = ""
    
    # Override tracking
    exposed_joins: set[str] = Field(default_factory=set)
    active_overrides: dict[str, Any] = Field(default_factory=dict)
    
    # Run metadata
    run_id: str = ""
    timestamp: Optional[str] = None
    trial_index: Optional[int] = None  # For sweep operations


# Legacy Ctx class - deprecated, kept for backwards compatibility
class Ctx:
    """Legacy context object - DEPRECATED.
    
    This is kept for backwards compatibility but should not be used.
    New code should use State and Context separately.
    """
    def __init__(self, state: PipelineState | None = None):
        self.state = state or PipelineState()
        self.metadata = self.state.metadata
        self.artifacts = self.state.artifacts
        self._current_stage: str | None = None


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

    def execute(self, state: PipelineState, ctx: Context) -> None:
        """Execute the stage function with state and context.
        
        Clean execution pattern:
        - Stages receive both state (mutable) and context (read-only)
        - They modify state directly, read context for execution info
        - No return values needed
        """
        try:
            # Check function signature to determine calling convention
            import inspect
            sig = inspect.signature(self.func)
            params = list(sig.parameters.keys())
            
            # Determine how to call the function based on its signature
            if len(params) == 0:
                # No-arg function (legacy or simple)
                self.func()
            elif len(params) == 1:
                # Single arg - assume it wants state
                self.func(state)
            else:
                # Two or more args - pass state and context
                self.func(state, ctx)
                
        except Exception as e:
            # Add context about which stage failed
            stage_type = "I/O operation" if self.stage_type == "io" else "Stage"
            error_msg = f"{stage_type} '{self.name}' failed: {str(e)}"
            
            # Re-raise with context
            if isinstance(e, (TypeError, ValueError, KeyError, AttributeError, IndexError)):
                raise type(e)(error_msg) from e
            elif isinstance(e, IOError):
                raise  # IOError already has context
            else:
                raise RuntimeError(error_msg) from e


class Pipeline:
    """Stateful Pipeline for composing stages that act on shared state.

    Key features:
    - Initialized with a state object that all stages share
    - Stages act directly on state (no explicit returns)
    - Method chaining API for building computation graphs
    - Join points for variation and A/B testing
    
    Usage patterns:
    
    1. Class-based with state:
       ```python
       class TrainingState(PipelineState):
           model: Any = None
           optimizer: str = "Adam"
       
       state = TrainingState()
       pipeline = Pipeline(state)
       pipeline.stage("train", lambda s: setattr(s, 'model', 'trained'))
       ```
       
    2. Direct state manipulation:
       ```python
       def train_model(state: PipelineState):
           state.metadata['epochs'] = 10
           # Direct mutation, no return
       
       pipeline.stage("train", train_model)
       ```
    """

    def __init__(self, state: PipelineState | None = None, name: str = ""):
        """Initialize pipeline with typed state.
        
        Args:
            state: Typed state object (creates base PipelineState if None)
            name: Optional pipeline name for identification
        """
        self.state = state or PipelineState()
        self.stages: list[Stage] = []
        self.stage_names: set[str] = set()
        self._built = False
        self.name = name
        
        # Join point management
        self._required_joins: dict[str, dict] = {}
        self._provided_joins: dict[str, Pipeline] = {}
        
        # Override tracking for deep overrides
        self._exposed: set[str] = set()  # Components marked as overrideable
        self._overrides: dict[str, Callable | Pipeline] = {}  # Active overrides
        self._parent_path: str = ""  # For building pipeline paths

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
        def execute_sub(state: PipelineState, ctx: Context) -> None:
            # Sub-pipeline shares state, gets nested context
            sub_ctx = ctx.model_copy(deep=True)
            sub_ctx.pipeline_path = f"{ctx.pipeline_path}.{name}"
            sub.state = state  # Share state with sub-pipeline
            sub.run(sub_ctx)
        
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

    def run(self, ctx: Context | None = None) -> PipelineState:
        """Execute the pipeline with clean state/context separation.

        Args:
            ctx: Optional execution context (created if not provided)

        Returns:
            The final state after all stages have executed
        """
        # Build if not already built
        if not self._built:
            self.build()

        # Create context if not provided
        if ctx is None:
            import datetime
            ctx = Context(
                current_stage="",
                stage_history=[],
                pipeline_path=self._parent_path or self.name,
                exposed_joins=self._exposed.copy(),
                active_overrides=self._overrides.copy(),
                timestamp=datetime.datetime.utcnow().isoformat()
            )

        # Execute stages in sequence with state and context
        for stage_obj in self.stages:
            stage_name = stage_obj.name
            
            # Update context for this stage
            ctx.current_stage = stage_name
            ctx.stage_history.append(stage_name)
            
            # Handle join stages with overrides
            if stage_name.startswith("join:"):
                join_name = stage_name[5:]  # Remove "join:" prefix
                if join_name in self._exposed and join_name in self._overrides:
                    # Execute override
                    override_func = self._overrides[join_name]
                    if isinstance(override_func, Pipeline):
                        # Sub-pipeline shares state, gets nested context
                        sub_ctx = ctx.model_copy(deep=True)
                        sub_ctx.pipeline_path = f"{ctx.pipeline_path}.{join_name}"
                        override_func.state = self.state
                        override_func.run(sub_ctx)
                    else:
                        # Simple function override
                        import inspect
                        sig = inspect.signature(override_func)
                        if len(sig.parameters) >= 2:
                            override_func(self.state, ctx)
                        else:
                            override_func(self.state)
                else:
                    # Execute join normally
                    stage_obj.execute(self.state, ctx)
            elif stage_name in self._exposed and stage_name in self._overrides:
                # Execute override for regular stage
                override_func = self._overrides[stage_name]
                if isinstance(override_func, Pipeline):
                    sub_ctx = ctx.model_copy(deep=True)
                    sub_ctx.pipeline_path = f"{ctx.pipeline_path}.{stage_name}"
                    override_func.state = self.state
                    override_func.run(sub_ctx)
                else:
                    import inspect
                    sig = inspect.signature(override_func)
                    if len(sig.parameters) >= 2:
                        override_func(self.state, ctx)
                    else:
                        override_func(self.state)
            else:
                # Execute stage normally
                stage_obj.execute(self.state, ctx)

            # Run checks if present - checks inspect state
            if hasattr(stage_obj, 'checks') and stage_obj.checks:
                from metta.sweep.axiom.checks import CheckLevel
                
                for check in stage_obj.checks:
                    passed, error_msg = check.check(self.state)
                    if not passed:
                        if check.level == CheckLevel.FAIL:
                            raise ValueError(f"Check failed at stage '{stage_obj.name}': {error_msg}")
                        else:
                            print(f"Warning at stage '{stage_obj.name}': {error_msg}")

            # Run hooks - receive state and context
            if hasattr(stage_obj, 'hooks'):
                for hook in stage_obj.hooks:
                    if callable(hook):
                        try:
                            import inspect
                            sig = inspect.signature(hook)
                            if len(sig.parameters) >= 2:
                                hook(self.state, ctx)
                            else:
                                hook(self.state)
                        except Exception as e:
                            hook_name = getattr(hook, '__name__', str(hook))
                            print(f"Warning: Hook '{hook_name}' at stage '{stage_obj.name}' failed: {e}")

        # Return the final state
        return self.state
    
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