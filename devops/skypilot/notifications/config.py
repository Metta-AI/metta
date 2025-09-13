#!/usr/bin/env python3


from dataclasses import dataclass
from enum import Enum


class NotificationState(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    UNKNOWN = "unknown"

    def to_github(self) -> str:
        # GitHub uses "success", "failure", "error", "pending"
        return {
            NotificationState.SUCCESS: "success",
            NotificationState.FAILURE: "failure",
            NotificationState.WARNING: "error",
            NotificationState.UNKNOWN: "error",
        }[self]

    def to_wandb(self) -> str:
        # W&B accepts "success" or "failure"
        return {
            NotificationState.SUCCESS: "success",
            NotificationState.FAILURE: "failure",
            NotificationState.WARNING: "failure",
            NotificationState.UNKNOWN: "failure",
        }[self]


@dataclass
class NotificationConfig:
    """Configuration for a notification."""

    emoji: str
    title: str
    description: str
    state: NotificationState = NotificationState.FAILURE
    discord: bool = True
    wandb: bool = True
    github: bool = True
