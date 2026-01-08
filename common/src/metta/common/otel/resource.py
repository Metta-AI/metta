from __future__ import annotations

import os
from typing import Optional


def parse_resource_attributes(raw: Optional[str]) -> dict[str, str]:
    if not raw:
        return {}
    attrs: dict[str, str] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key:
            attrs[key] = value.strip()
    return attrs


def format_resource_attributes(attrs: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in attrs.items())


def build_resource_attributes(service_name: Optional[str] = None, include_service_name: bool = True) -> dict[str, str]:
    attrs = parse_resource_attributes(os.environ.get("OTEL_RESOURCE_ATTRIBUTES"))

    if include_service_name:
        resolved_service_name = service_name or os.environ.get("OTEL_SERVICE_NAME")
        if resolved_service_name:
            attrs["service.name"] = resolved_service_name
        elif "service.name" not in attrs:
            attrs["service.name"] = "metta"
    else:
        attrs.pop("service.name", None)

    if "deployment.environment" not in attrs:
        deployment_env = os.environ.get("DEPLOYMENT_ENVIRONMENT") or os.environ.get("OTEL_DEPLOYMENT_ENVIRONMENT")
        attrs["deployment.environment"] = deployment_env or "development"

    if "service.version" not in attrs:
        version = os.environ.get("OTEL_SERVICE_VERSION") or os.environ.get("METTA_GIT_REF")
        attrs["service.version"] = version or "unknown"

    return attrs
