"""Shared constants and utilities for policy submission archives."""

from typing import Optional

from pydantic import BaseModel, Field

POLICY_SPEC_FILENAME = "policy_spec.json"


class SubmissionPolicySpec(BaseModel):
    """Policy specification as stored in submission archives.

    This is the serialized format written to POLICY_SPEC_FILENAME in submission zips.
    It extends the core PolicySpec fields with submission-specific options like setup_script.
    """

    class_path: str = Field(description="Fully qualified path to policy class")
    data_path: Optional[str] = Field(default=None, description="Relative path to policy data within archive")
    init_kwargs: dict = Field(default_factory=dict, description="Keyword arguments for policy initialization")
    setup_script: Optional[str] = Field(
        default=None,
        description="Relative path to a Python setup script to run once before loading the policy",
    )
