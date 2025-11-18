import functools
import logging
import sys

from ddtrace.trace import tracer

from metta.common.datadog.config import datadog_config

logger = logging.getLogger(__name__)


@functools.cache
def init_tracing():
    if datadog_config.DD_TRACE_ENABLED:
        if datadog_config.DD_LOGS_INJECTION:
            # Inject trace identifiers into stdlib logging records for log/trace correlation
            from ddtrace import patch
            from ddtrace.contrib.logging import TraceLogFilter

            patch(logging=True)

            dd_handler = logging.StreamHandler(sys.stdout)
            dd_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(name)s] %(levelname)s "
                    "[dd.service=%(dd.service)s dd.env=%(dd.env)s dd.version=%(dd.version)s "
                    "dd.trace_id=%(dd.trace_id)s dd.span_id=%(dd.span_id)s]: %(message)s"
                )
            )

            dd_handler.addFilter(TraceLogFilter())
            logging.getLogger().addHandler(dd_handler)
            logger.info(
                f"Datadog tracing enabled: service={datadog_config.DD_SERVICE}, "
                f"env={datadog_config.DD_ENV}, agent={datadog_config.DD_AGENT_HOST}"
            )
        tracer.enabled = True
    else:
        logger.info("Datadog tracing disabled")
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
