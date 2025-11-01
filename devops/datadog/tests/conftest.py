"""Pytest configuration for datadog tests."""

import os


def pytest_configure(config):
    """Configure pytest for datadog tests.

    Remove the OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE environment
    variable to suppress Datadog warnings about unsupported OpenTelemetry config.
    """
    # Remove the unsupported OTEL config that triggers Datadog warnings
    os.environ.pop("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE", None)
