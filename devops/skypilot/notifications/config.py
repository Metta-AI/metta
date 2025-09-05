#!/usr/bin/env python3


from dataclasses import dataclass


@dataclass
class NotificationConfig:
    """Configuration for a notification."""

    emoji: str
    title: str
    description: str
    discord: bool = True
    wandb: bool = True
    github: bool = True
    github_state: str = "failure"
