import functools
import os
import sys

from ddtrace.trace import tracer

from metta.common.datadog.config import datadog_config


@functools.cache
def init_tracing():
    # Debug: print env vars to verify they're set
    dd_trace_enabled_env = os.environ.get("DD_TRACE_ENABLED", "not set")
    print(f"[DD Debug] DD_TRACE_ENABLED env var: {dd_trace_enabled_env}", file=sys.stderr, flush=True)
    print(f"[DD Debug] DD_TRACE_ENABLED config value: {datadog_config.DD_TRACE_ENABLED}", file=sys.stderr, flush=True)

    if datadog_config.DD_TRACE_ENABLED:
        msg = (
            f"Datadog tracing enabled: service={datadog_config.DD_SERVICE}, "
            f"env={datadog_config.DD_ENV}, agent={datadog_config.DD_AGENT_HOST}:{datadog_config.DD_TRACE_AGENT_PORT}"
        )
        print(msg, file=sys.stderr, flush=True)
        tracer.enabled = True
    else:
        print("Datadog tracing disabled", file=sys.stderr, flush=True)
        tracer.enabled = False


def trace(name: str):
    init_tracing()

    def decorator(func):
        import asyncio

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with tracer.trace(name):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with tracer.trace(name):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
