import functools
import logging
import os
import traceback
from contextlib import contextmanager
from typing import Any, Callable, ParamSpec, TypeVar

from ddtrace import patch
from ddtrace.trace import tracer

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
        enabled = os.getenv("DD_TRACE_ENABLED", "false").lower() == "true"

    if not enabled:
        logger.info("Datadog tracing disabled")
        tracer.enabled = False
        return

    # Configure tracer (without deprecated hostname/port parameters)
    # Agent connection is configured via environment variables:
    # - DD_TRACE_AGENT_URL or DD_AGENT_HOST + DD_AGENT_PORT
    tracer.configure()

    # Set global tags
    tags = {}

    if env:
        tags["env"] = env
    elif os.getenv("DD_ENV"):
        tags["env"] = os.getenv("DD_ENV")

    if version:
        tags["version"] = version
    elif os.getenv("DD_VERSION"):
        tags["version"] = os.getenv("DD_VERSION")

    if service_name:
        tags["service"] = service_name
    elif os.getenv("DD_SERVICE"):
        tags["service"] = os.getenv("DD_SERVICE")

    if tags:
        tracer.set_tags(tags)

    # Auto-patch common libraries
    patch(httpx=True)

    # Get agent connection info from environment
    agent_url = os.getenv("DD_TRACE_AGENT_URL")
    if agent_url:
        agent_info = agent_url
    else:
        agent_host = os.getenv("DD_AGENT_HOST", "localhost")
        agent_port = os.getenv("DD_TRACE_AGENT_PORT", "8126")
        agent_info = f"{agent_host}:{agent_port}"

    logger.info(
        f"Datadog tracing configured: service={service_name}, "
        f"env={env or os.getenv('DD_ENV', 'not-set')}, "
        f"agent={agent_info}"
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
            for k, v in tags.items():
                span.set_tag(k, v)
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

                result = await func(*args, **kwargs)
                return result

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

                result = func(*args, **kwargs)
                return result

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
        for k, v in tags.items():
            span.set_tag(k, v)


def set_span_error(error: Exception) -> None:
    """Mark current span as error with exception details."""
    span = tracer.current_span()
    if span:
        span.set_tag("error", True)
        span.set_tag("error.msg", str(error))
        span.set_tag("error.type", type(error).__name__)
        span.set_tag("error.stack", traceback.format_exc())
