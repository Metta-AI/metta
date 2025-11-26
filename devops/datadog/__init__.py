"""
Datadog metric collectors for Metta infra health.

This package hosts:
- Collector implementations (CI, training, eval)
- Shared models for metric samples
- CLI entrypoints used by Kubernetes CronJobs
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
