"""
Task Management Pydantic models for MCP server.

This module contains models for task creation, assignment, updates,
and worker management in the evaluation system.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskList(BaseModel):
    """List of tasks."""

    tasks: list[dict[str, str | int | float | bool | None]] = Field(description="List of task objects")
    total_count: int = Field(description="Total number of tasks")


class TaskCreationResult(BaseModel):
    """Result of task creation."""

    task_id: str = Field(description="Unique identifier for the created task")
    policy_id: str = Field(description="Policy ID associated with the task")
    sim_suite: str = Field(description="Simulation suite for the task")
    status: str = Field(description="Initial task status")
    created_at: str = Field(description="ISO timestamp when task was created")


class TaskAssignment(BaseModel):
    """Task assignment information."""

    task_id: str = Field(description="Unique task identifier")
    assignee: str = Field(description="Worker/assignee identifier")
    assigned_at: str = Field(description="ISO timestamp when task was assigned")
    policy_id: str = Field(description="Policy ID for the assigned task")
    status: str = Field(description="Current task status")


class TaskClaimResult(BaseModel):
    """Result of claiming tasks."""

    claimed_tasks: list[str] = Field(description="List of successfully claimed task IDs")
    assignee: str = Field(description="Worker who claimed the tasks")
    total_claimed: int = Field(description="Number of tasks successfully claimed")


class TaskUpdateResult(BaseModel):
    """Result of task status updates."""

    updated_tasks: list[str] = Field(description="List of task IDs that were successfully updated")
    total_updated: int = Field(description="Number of tasks successfully updated")
    errors: list[str] = Field(description="List of any errors encountered during updates")


class WorkerGitInfo(BaseModel):
    """Git hash information for workers."""

    worker_git_hashes: dict[str, str] = Field(description="Mapping of worker names to their git commit hashes")
