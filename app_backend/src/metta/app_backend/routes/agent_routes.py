"""Routes for Cogweb agent storage configuration."""

from __future__ import annotations

from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.config import cogweb_agent_bucket_region, cogweb_agent_bucket_uri
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.route_logger import timed_route


class AgentBucketResponse(BaseModel):
    """Response payload describing where Cogweb agents should be stored."""

    bucket: str
    prefix: str
    uri: str
    region: str


def _parse_bucket_components(uri: str) -> tuple[str, str]:
    """Split an S3 URI into bucket and prefix components."""

    parsed = urlparse(uri)

    if parsed.scheme == "s3" and parsed.netloc:
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return bucket, prefix

    raise ValueError(f"Invalid S3 URI provided for Cogweb agent bucket: {uri}")


def _build_uri(bucket: str, prefix: str) -> str:
    """Construct an S3 URI from bucket and optional prefix."""

    if prefix:
        return f"s3://{bucket}/{prefix}"
    return f"s3://{bucket}"


def create_agent_router(metta_repo: MettaRepo) -> APIRouter:
    """Create the agent configuration router."""

    router = APIRouter(prefix="/agents", tags=["agents"])
    user_or_token = Depends(create_user_or_token_dependency(metta_repo))

    @router.get("/bucket", response_model=AgentBucketResponse)
    @timed_route("get_agent_bucket")
    async def get_agent_bucket(user: str = user_or_token) -> AgentBucketResponse:  # type: ignore[reportUnusedFunction]
        try:
            bucket, prefix = _parse_bucket_components(cogweb_agent_bucket_uri)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not bucket:
            raise HTTPException(status_code=500, detail="Cogweb agent bucket is not configured")

        return AgentBucketResponse(
            bucket=bucket,
            prefix=prefix,
            uri=_build_uri(bucket, prefix),
            region=cogweb_agent_bucket_region,
        )

    return router
