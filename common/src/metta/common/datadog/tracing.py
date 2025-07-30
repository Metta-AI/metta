import functools
import logging
import os
import traceback
from contextlib import contextmanager
from typing import Any, Callable, ParamSpec, TypeVar

from ddtrace import patch, tracer

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def configure_datadog_tracing(
    service_name: str = "metta-eval",
    env: str | None = None,
    version: str | None = None,
    enabled: bool | None = None,
) -> None:
    """
    Configure Datadog tracing.

    Args:
        service_name: Service name for traces (default: metta-eval)
        env: Environment name (production, staging, etc)
        version: Service version (e.g., git hash)
        enabled: Whether tracing is enabled (defaults to DD_TRACE_ENABLED env var)
    """
    if enabled is None:
        enabled = os.getenv("DD_TRACE_ENABLED", "true").lower() == "true"

    if not enabled:
        logger.info("Datadog tracing disabled")
        tracer.enabled = False
        return

    # Configure tracer
    tracer.configure(
        hostname=os.getenv("DD_AGENT_HOST", "localhost"),
        port=int(os.getenv("DD_TRACE_AGENT_PORT", "8126")),
        debug=os.getenv("DD_TRACE_DEBUG", "false").lower() == "true",
        enabled=enabled,
    )

    # Set global tags
    if env or os.getenv("DD_ENV"):
        tracer.set_tags({"env": env or os.getenv("DD_ENV")})

    if version or os.getenv("DD_VERSION"):
        tracer.set_tags({"version": version or os.getenv("DD_VERSION")})

    if service_name or os.getenv("DD_SERVICE"):
        tracer.set_tags({"service": service_name or os.getenv("DD_SERVICE")})

    # Auto-patch common libraries
    patch(httpx=True)

    logger.info(
        f"Datadog tracing configured: service={service_name}, "
        f"env={env or os.getenv('DD_ENV', 'not-set')}, "
        f"agent={os.getenv('DD_AGENT_HOST', 'localhost')}:{os.getenv('DD_TRACE_AGENT_PORT', '8126')}"
    )


@contextmanager
def traced_span(
    operation_name: str,
    resource: str | None = None,
    service: str | None = None,
    tags: dict[str, Any] | None = None,
):
    """
    Context manager for creating a traced span.

    Args:
        operation_name: Name of the operation being traced
        resource: Resource name (e.g., specific task ID)
        service: Override service name for this span
        tags: Additional tags to add to the span

    Example:
        with traced_span("eval.task.process", resource=f"task_{task_id}", tags={"task_id": task_id}):
            process_task()
    """
    with tracer.trace(operation_name, resource=resource, service=service) as span:
        if tags:
            span.set_tags(tags)
        try:
            yield span
        except Exception as e:
            span.set_tag("error", True)
            span.set_tag("error.msg", str(e))
            span.set_tag("error.type", type(e).__name__)
            raise


def trace_method(
    operation_name: str | None = None,
    resource_from: str | None = None,
    tags: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing methods/functions.

    Args:
        operation_name: Override operation name (defaults to module.function)
        resource_from: Parameter name to use as resource (e.g., "task_id")
        tags: Static tags to add to all spans

    Example:
        @trace_method(resource_from="task_id", tags={"component": "worker"})
        async def process_task(self, task_id: str):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            resource = None
            if resource_from:
                # Try to get resource from kwargs first, then args
                if resource_from in kwargs:
                    resource = str(kwargs[resource_from])
                else:
                    # For methods, skip 'self' in args
                    func_args = list(args[1:] if args and hasattr(args[0], "__class__") else args)
                    param_names = list(func.__code__.co_varnames[1 : func.__code__.co_argcount])
                    if resource_from in param_names:
                        idx = param_names.index(resource_from)
                        if idx < len(func_args):
                            resource = str(func_args[idx])

            with traced_span(op_name, resource=resource, tags=tags) as span:
                # Add method-specific tags
                if args and hasattr(args[0], "__class__"):
                    span.set_tag("class", args[0].__class__.__name__)

                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            resource = None
            if resource_from:
                if resource_from in kwargs:
                    resource = str(kwargs[resource_from])
                else:
                    func_args = list(args[1:] if args and hasattr(args[0], "__class__") else args)
                    param_names = list(func.__code__.co_varnames[1 : func.__code__.co_argcount])
                    if resource_from in param_names:
                        idx = param_names.index(resource_from)
                        if idx < len(func_args):
                            resource = str(func_args[idx])

            with traced_span(op_name, resource=resource, tags=tags) as span:
                if args and hasattr(args[0], "__class__"):
                    span.set_tag("class", args[0].__class__.__name__)

                return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def add_span_tags(tags: dict[str, Any]) -> None:
    """Add tags to the current active span."""
    span = tracer.current_span()
    if span:
        span.set_tags(tags)


def set_span_error(error: Exception) -> None:
    """Mark current span as error with exception details."""
    span = tracer.current_span()
    if span:
        span.set_tag("error", True)
        span.set_tag("error.msg", str(error))
        span.set_tag("error.type", type(error).__name__)
        span.set_tag("error.stack", traceback.format_exc())
