"""
Configuration for Observatory MCP Server

Handles server configuration, authentication settings, and default values
for connecting to the Metta Observatory backend and external services.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse


@dataclass
class ObservatoryMCPConfig:
    """Configuration for the Observatory MCP Server."""

    # Server configuration
    server_name: str = "observatory-mcp"
    version: str = "0.1.0"

    # Backend API configuration
    backend_url: str = field(
        default_factory=lambda: os.getenv("METTA_MCP_BACKEND_URL", "http://localhost:8000")
    )

    # Authentication configuration
    machine_token: Optional[str] = field(
        default_factory=lambda: os.getenv("METTA_MCP_MACHINE_TOKEN")
    )

    # AWS configuration
    aws_profile: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_PROFILE")
    )
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("METTA_S3_BUCKET", "softmax-public")
    )

    # W&B configuration
    wandb_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_API_KEY")
    )
    wandb_entity: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_ENTITY")
    )
    wandb_project: Optional[str] = field(
        default_factory=lambda: os.getenv("WANDB_PROJECT")
    )

    # Skypilot configuration
    skypilot_url: Optional[str] = field(
        default_factory=lambda: os.getenv("METTA_SKYPILOT_URL")
    )

    # Logging configuration
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> "ObservatoryMCPConfig":
        """Create configuration from environment variables.

        Returns:
            ObservatoryMCPConfig instance with values from environment variables
        """
        return cls(
            backend_url=os.getenv("METTA_MCP_BACKEND_URL", "http://localhost:8000"),
            machine_token=os.getenv("METTA_MCP_MACHINE_TOKEN"),
            aws_profile=os.getenv("AWS_PROFILE"),
            s3_bucket=os.getenv("METTA_S3_BUCKET", "softmax-public"),
            wandb_api_key=os.getenv("WANDB_API_KEY"),
            wandb_entity=os.getenv("WANDB_ENTITY"),
            wandb_project=os.getenv("WANDB_PROJECT"),
            skypilot_url=os.getenv("METTA_SKYPILOT_URL"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> List[str]:
        """Validate the configuration and return any errors.

        Returns:
            List of error messages. Empty list if configuration is valid.
        """
        errors = []
        if not self.backend_url:
            errors.append("METTA_MCP_BACKEND_URL is required but not set.")
        else:
            try:
                parsed = urlparse(self.backend_url)
                if not parsed.scheme or not parsed.netloc:
                    errors.append(
                        f"Invalid backend URL format: {self.backend_url}. "
                        "Expected format: http://host:port or https://host:port"
                    )
            except Exception as e:
                errors.append(f"Error parsing backend URL: {e}")

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(
                f"Invalid log level: {self.log_level}. "
                f"Must be one of: {', '.join(valid_log_levels)}"
            )

        if self.s3_bucket:
            if not (3 <= len(self.s3_bucket) <= 63):
                errors.append(
                    f"Invalid S3 bucket name: {self.s3_bucket}. "
                    "Bucket names must be between 3 and 63 characters."
                )

        return errors

    def is_backend_configured(self) -> bool:
        """Check if backend is properly configured."""
        return bool(self.backend_url) and len(self.validate()) == 0

    def is_aws_configured(self) -> bool:
        """Check if AWS is configured.

        Always returns True to allow boto3 to attempt initialization using either:
        - AWS_PROFILE environment variable (if set), or
        - Default AWS credentials (from ~/.aws/credentials, IAM role, etc.)

        Returns:
            Always True - initialization will be attempted regardless
        """
        return True  # Always try - boto3 can use default credentials or profile

    def is_wandb_configured(self) -> bool:
        """Check if W&B is configured."""
        return bool(self.wandb_api_key)

    def is_authenticated(self) -> bool:
        """Check if authentication token is available."""
        return bool(self.machine_token)

