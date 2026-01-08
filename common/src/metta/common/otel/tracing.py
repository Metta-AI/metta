from __future__ import annotations

import asyncio
import atexit
import logging
import os
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

T = TypeVar("T")

_initialized = False
_tracer_provider: Optional[TracerProvider] = None


def trace(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        tracer = otel_trace.get_tracer(func.__module__)
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(name):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def init_otel_tracing(service_name: str) -> None:
    """
    Initialize OpenTelemetry tracing for this process.

    This module configures tracing only (TracerProvider + optional OTLP span export).
    It does not configure logging, log correlation, or log export.

    Tracing behavior is controlled via environment variables:
      - OTEL_TRACES_ENABLED: set to "true" to export spans (default: false)
      - OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: where to send spans
      - OTEL_EXPORTER_OTLP_PROTOCOL, OTEL_EXPORTER_OTLP_HEADERS, etc.: exporter options

    If you want trace/span IDs injected into Python logs and/or logs exported via OTLP,
    run the application with OpenTelemetry Python auto-instrumentation and its logging env vars
    (e.g., OTEL_PYTHON_LOG_CORRELATION, OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED).
    """
    global _initialized, _tracer_provider
    if _initialized:
        return
    _initialized = True

    resource_attributes = {"service.name": service_name}
    resource = Resource.create(resource_attributes)

    tracer_provider = TracerProvider(resource=resource)
    _tracer_provider = tracer_provider
    otel_trace.set_tracer_provider(tracer_provider)

    if os.getenv("OTEL_TRACES_ENABLED", "").lower() in ("1", "true"):
        try:
            # Exporter reads standard env vars for endpoint/protocol/headers/etc.
            tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to configure trace exporter: {e}")

    atexit.register(_shutdown)


def _shutdown() -> None:
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
