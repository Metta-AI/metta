import functools
import logging

import ddtrace.trace

import metta.common.datadog.config

logger = logging.getLogger(__name__)


@functools.cache
def init_tracing():
    if metta.common.datadog.config.datadog_config.DD_TRACE_ENABLED:
        dd_config = metta.common.datadog.config.datadog_config
        logger.info(
            "Datadog tracing enabled: "
            f"service={dd_config.DD_SERVICE}, env={dd_config.DD_ENV}, agent={dd_config.DD_AGENT_HOST}"
        )
        ddtrace.trace.tracer.enabled = True
    else:
        logger.info("Datadog tracing disabled")
        ddtrace.trace.tracer.enabled = False


def trace(name: str):
    def decorator(func):
        import asyncio

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                with ddtrace.trace.tracer.trace(name):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                with ddtrace.trace.tracer.trace(name):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
