from __future__ import annotations

import asyncio
import atexit
import logging
import os
from functools import wraps
from typing import Callable, Optional, TypeVar

from opentelemetry import propagate
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from metta.common.otel.resource import build_resource_attributes

T = TypeVar("T")

_initialized = False
_tracer_provider: Optional[TracerProvider] = None
_logger_provider: Optional[LoggerProvider] = None


class TraceContextFilter(logging.Filter):
    def __init__(self, service: Optional[str], env: Optional[str], version: Optional[str]) -> None:
        super().__init__()
        self._service = service
        self._env = env
        self._version = version

    def filter(self, record: logging.LogRecord) -> bool:
        span = otel_trace.get_current_span()
        span_context = span.get_span_context()
        if span_context and span_context.is_valid:
            record.trace_id = format(span_context.trace_id, "032x")
            record.span_id = format(span_context.span_id, "016x")
        else:
            record.trace_id = None
            record.span_id = None
        if self._service is not None:
            record.service = self._service
        if self._env is not None:
            record.env = self._env
        if self._version is not None:
            record.version = self._version
        return True


def get_tracer(name: Optional[str] = None) -> otel_trace.Tracer:
    return otel_trace.get_tracer(name or __name__)


def trace(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer(func.__module__)
                with tracer.start_as_current_span(name):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(name):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def trace_from_carrier(
    name: str,
    carrier_getter: Callable[..., Optional[dict[str, str]]],
    **span_kwargs: object,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                carrier = carrier_getter(*args, **kwargs) or {}
                parent_context = propagate.extract(carrier)
                tracer = get_tracer(func.__module__)
                with tracer.start_as_current_span(name, context=parent_context, **span_kwargs):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            carrier = carrier_getter(*args, **kwargs) or {}
            parent_context = propagate.extract(carrier)
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(name, context=parent_context, **span_kwargs):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def init_tracing(service_name: Optional[str] = None) -> None:
    global _initialized, _tracer_provider, _logger_provider
    if _initialized:
        return
    _initialized = True

    resource_attributes = build_resource_attributes(service_name=service_name)
    resource = Resource.create(resource_attributes)

    tracer_provider = TracerProvider(resource=resource)
    _tracer_provider = tracer_provider
    otel_trace.set_tracer_provider(tracer_provider)

    traces_exporter = os.environ.get("OTEL_TRACES_EXPORTER", "otlp").lower()
    if traces_exporter != "none":
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    _install_log_enricher(resource_attributes)
    _configure_log_exporter(resource)

    atexit.register(_shutdown)


def _install_log_enricher(resource_attributes: dict[str, str]) -> None:
    root_logger = logging.getLogger()
    existing_filters = [f for f in root_logger.filters if isinstance(f, TraceContextFilter)]
    if existing_filters:
        trace_filter = existing_filters[0]
    else:
        trace_filter = TraceContextFilter(
            service=resource_attributes.get("service.name"),
            env=resource_attributes.get("deployment.environment"),
            version=resource_attributes.get("service.version"),
        )
        root_logger.addFilter(trace_filter)

    for handler in root_logger.handlers:
        if any(isinstance(f, TraceContextFilter) for f in handler.filters):
            continue
        handler.addFilter(trace_filter)


def _configure_log_exporter(resource: Resource) -> None:
    global _logger_provider
    logs_exporter = os.environ.get("OTEL_LOGS_EXPORTER", "none").lower()
    if logs_exporter != "otlp":
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint)))
    _logger_provider = logger_provider

    from opentelemetry._logs import set_logger_provider

    set_logger_provider(logger_provider)
    logging_handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    logging.getLogger().addHandler(logging_handler)


def _shutdown() -> None:
    if _logger_provider is not None:
        _logger_provider.shutdown()
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
