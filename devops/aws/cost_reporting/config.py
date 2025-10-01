from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CostReportingSettings(BaseSettings):
    """Configuration for AWS cost reporting.

    Values are loaded from environment variables with sensible defaults. This
    settings object is safe to construct in CLIs and modules.
    """

    model_config = SettingsConfigDict(env_prefix="AWS_COST_", extra="ignore")

    # AWS + account scope
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    billing_account_id: str | None = Field(
        default=None, description="Payer/management account ID for org-wide cost access."
    )
    assume_role_name: str | None = Field(default=None, description="Role name to assume in linked accounts if needed.")
    linked_account_ids: list[str] = Field(
        default_factory=list, description="Explicit linked account IDs (optional; use CE LINKED_ACCOUNT otherwise)."
    )

    # Storage
    s3_bucket: str | None = Field(
        default=None, description="S3 bucket for Parquet/artefacts. If unset, falls back to local_dir."
    )
    s3_prefix: str = Field(default="aws_cost", description="S3 key prefix top-level directory.")
    local_dir: Path = Field(default=Path("devops/aws/cost_reporting/data"))

    # Cost Explorer query defaults
    granularity: Literal["DAILY", "MONTHLY"] = "DAILY"
    metrics: list[str] = Field(
        default_factory=lambda: ["UnblendedCost"]
    )  # ["UnblendedCost", "AmortizedCost", "BlendedCost"]
    tag_keys: list[str] = Field(
        default_factory=list, description="Cost allocation tag keys to include, e.g. ['Project','Team']."
    )

    # Time window (optional; CLI can override). If unset, collectors must supply.
    start_date: date | None = None
    end_date: date | None = None

    # Notifications (optional)
    slack_webhook_url: str | None = None
    email_from: str | None = None
    email_to: str | None = None

    def data_root(self) -> Path:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        return self.local_dir
