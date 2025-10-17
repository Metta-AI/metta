import os
from typing import Final

from metta.common.util.constants import METTA_AWS_REGION, SOFTMAX_S3_POLICY_PREFIX

stats_db_uri = os.getenv("STATS_DB_URI", "postgres://postgres:password@127.0.0.1/postgres")
debug_user_email = os.getenv("DEBUG_USER_EMAIL")

host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", "8000"))

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

run_leaderboard_updater = os.getenv("RUN_LEADERBOARD_UPDATER", "true") == "true"


def _normalize_s3_uri(value: str) -> str:
    """Normalize potential S3 bucket URIs.

    Accepts bare bucket names ("my-bucket"), bucket plus prefix ("my-bucket/prefix"),
    or fully-qualified URIs ("s3://my-bucket/prefix"). Returns a normalized
    ``s3://bucket[/prefix]`` string without a trailing slash.
    """

    trimmed = value.strip()
    if not trimmed:
        return SOFTMAX_S3_POLICY_PREFIX

    if trimmed.startswith("s3://"):
        normalized = trimmed.rstrip("/")
    else:
        normalized = f"s3://{trimmed.lstrip('/')}".rstrip("/")

    return normalized


_raw_agent_bucket = os.getenv("COGWEB_AGENT_BUCKET_URI") or os.getenv("COGWEB_AGENT_BUCKET")
cogweb_agent_bucket_uri: Final[str] = (
    _normalize_s3_uri(_raw_agent_bucket) if _raw_agent_bucket else SOFTMAX_S3_POLICY_PREFIX
)
cogweb_agent_bucket_region: Final[str] = os.getenv("COGWEB_AGENT_BUCKET_REGION", METTA_AWS_REGION)
