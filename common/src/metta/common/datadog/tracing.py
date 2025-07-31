import sys

from ddtrace.trace import tracer

from metta.common.datadog.config import datadog_config


def init_tracing():
    if datadog_config.DD_TRACE_ENABLED:
        msg = (
            f"Datadog tracing enabled: service={datadog_config.DD_SERVICE}, "
            f"env={datadog_config.DD_ENV}, agent={datadog_config.DD_AGENT_HOST}"
        )
        print(msg, file=sys.stderr, flush=True)
        tracer.enabled = True
    else:
        print("Datadog tracing disabled", file=sys.stderr, flush=True)
        tracer.enabled = False


def trace(name: str):
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
