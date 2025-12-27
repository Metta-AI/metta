"""Shared constants and utilities for policy submission archives."""

import os
import zipfile
from pathlib import Path
from typing import Iterable, Optional

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


def write_submission_zip(
    zip_path: Path,
    *,
    submission_spec: SubmissionPolicySpec,
    include_files: Iterable[Path],
) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(data=submission_spec.model_dump_json(), zinfo_or_arcname=POLICY_SPEC_FILENAME)

        for file_path in include_files:
            if file_path.is_dir():
                for root, _, files in os.walk(file_path):
                    for file in files:
                        file_full_path = Path(root) / file
                        zipf.write(file_full_path, arcname=file_full_path)
            else:
                zipf.write(file_path, arcname=file_path)


def write_policy_bundle_zip(bundle_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, arcname=file_path.relative_to(bundle_dir))
