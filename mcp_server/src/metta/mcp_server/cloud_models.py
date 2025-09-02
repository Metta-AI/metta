"""
Cloud provider Pydantic models for MCP server.

This module contains models for AWS S3, Weights & Biases, Skypilot,
and other cloud services used in the MCP server.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# AWS S3 Models
class S3ObjectList(BaseModel):
    """List of S3 objects."""

    objects: list[str] = Field(description="List of S3 object keys")
    bucket: str = Field(description="S3 bucket name")
    prefix: str | None = Field(description="Prefix used to filter objects")


class S3PrefixList(BaseModel):
    """List of S3 prefixes (directories)."""

    prefixes: list[str] = Field(description="List of S3 common prefixes ending with '/'")
    bucket: str = Field(description="S3 bucket name")
    parent_prefix: str | None = Field(description="Parent prefix searched under")


class S3ObjectMetadata(BaseModel):
    """S3 object metadata."""

    content_length: int = Field(description="Size of the object in bytes")
    content_type: str = Field(description="MIME type of the object")
    etag: str = Field(description="Entity tag of the object")
    last_modified: str = Field(description="ISO timestamp when object was last modified")
    storage_class: str | None = Field(description="S3 storage class")
    metadata: dict[str, str] = Field(description="Custom metadata key-value pairs")


# Weights & Biases Models
class WandbRun(BaseModel):
    """Single Weights & Biases run object."""

    id: str = Field(description="W&B run ID")
    name: str = Field(description="W&B run name")
    state: str = Field(description="Run state (running, finished, failed, etc.)")
    url: str = Field(description="URL to the W&B run page")


class WandbRunList(BaseModel):
    """List of Weights & Biases runs."""

    runs: list[dict[str, str | int | float | bool | None]] = Field(description="List of W&B run objects")
    project: str = Field(description="W&B project name")
    entity: str = Field(description="W&B entity/organization name")


class WandbError(BaseModel):
    """Wandb error response."""

    error: str = Field(description="Error message")
    status: str = Field(description="Error status")


# Skypilot Models
class SkypilotStatus(BaseModel):
    """Skypilot job status information."""

    status_output: str = Field(description="Raw output from 'sky status --verbose' command")
    success: bool = Field(description="Whether the command executed successfully")
