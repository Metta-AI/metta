import tempfile
import uuid
from typing import Annotated, Any, Optional

import aioboto3
import duckdb
from fastapi import APIRouter, Body, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from metta.app_backend.auth import UserOrToken
from metta.app_backend.metta_repo import (
    EpisodeWithTags,
    MettaRepo,
    PolicyVersionWithName,
    PublicPolicyVersionRow,
)
from metta.app_backend.route_logger import timed_route

OBSERVATORY_S3_BUCKET = "observatory-private"


# Request/Response Models
class UUIDResponse(BaseModel):
    id: uuid.UUID


class PolicyCreate(BaseModel):
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    is_system_policy: bool = False


class PolicyVersionCreate(BaseModel):
    policy_spec: dict[str, Any] = Field(default_factory=dict)
    git_hash: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class EpisodeCreate(BaseModel):
    agent_policy_versions: dict[int, uuid.UUID]
    # agent_id -> metric_name -> metric_value
    agent_metrics: dict[int, dict[str, float]]
    primary_policy_id: uuid.UUID | None = None
    replay_url: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    eval_task_id: uuid.UUID | None = None
    tags: list[tuple[str, str]] = Field(default_factory=list)
    thumbnail_url: str | None = None


class EpisodeResponse(BaseModel):
    id: uuid.UUID


class BulkEpisodeUploadResponse(BaseModel):
    """Response for bulk episode upload."""

    episodes_created: int
    duckdb_s3_uri: str


class PresignedUploadUrlResponse(BaseModel):
    """Response containing presigned URL for direct S3 upload."""

    upload_url: str
    s3_key: str
    upload_id: uuid.UUID


class CompleteBulkUploadRequest(BaseModel):
    """Request to complete a bulk upload."""

    upload_id: uuid.UUID


class CompletePolicySubmitRequest(BaseModel):
    """Request to complete a policy submission after uploading to S3."""

    upload_id: uuid.UUID
    name: str


class MyPolicyVersionsResponse(BaseModel):
    entries: list[PublicPolicyVersionRow]


class EpisodeQueryRequest(BaseModel):
    primary_policy_version_ids: Optional[list[uuid.UUID]] = None
    episode_ids: Optional[list[uuid.UUID]] = None
    tag_filters: Optional[dict[str, Optional[list[str]]]] = None
    limit: Optional[int] = 200
    offset: int = 0


class EpisodeQueryResponse(BaseModel):
    episodes: list[EpisodeWithTags]


class PolicyVersionsResponse(BaseModel):
    entries: list[PublicPolicyVersionRow]
    total_count: int


def create_stats_router(stats_repo: MettaRepo) -> APIRouter:
    """Create a stats router with the given StatsRepo instance."""
    router = APIRouter(prefix="/stats", tags=["stats"])

    async def _create_policy_version_from_s3_key(name: str, user_id: str, s3_key: str) -> UUIDResponse:
        s3_path = f"s3://{OBSERVATORY_S3_BUCKET}/{s3_key}"
        policy_id = await stats_repo.upsert_policy(name=name, user_id=user_id, attributes={})
        policy_version_id = await stats_repo.create_policy_version(
            policy_id=policy_id,
            s3_path=s3_path,
            git_hash=None,
            policy_spec={},
            attributes={},
        )
        return UUIDResponse(id=policy_version_id)

    @router.post("/policies")
    @timed_route("create_policy")
    async def upsert_policy(policy: PolicyCreate, user: UserOrToken) -> UUIDResponse:
        """Create a new policy."""
        if policy.is_system_policy:
            user_id = "system"
        else:
            user_id = user

        try:
            policy_id = await stats_repo.upsert_policy(name=policy.name, user_id=user_id, attributes=policy.attributes)
            return UUIDResponse(id=policy_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create policy: {str(e)}") from e

    @router.post("/policies/{policy_id_str}/versions")
    @timed_route("create_policy_version")
    async def create_policy_version(
        policy_id_str: str, policy_version: PolicyVersionCreate, user: UserOrToken
    ) -> UUIDResponse:
        """Create a new policy version."""
        try:
            policy_id = uuid.UUID(policy_id_str)
            policy_version_id = await stats_repo.create_policy_version(
                policy_id=policy_id,
                s3_path=None,
                git_hash=policy_version.git_hash,
                policy_spec=policy_version.policy_spec,
                attributes=policy_version.attributes,
            )
            return UUIDResponse(id=policy_version_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create policy version: {str(e)}") from e

    @router.get("/policies/versions/{policy_version_id_str}")
    @timed_route("get_policy_version")
    async def get_policy_version(policy_version_id_str: str) -> PolicyVersionWithName:
        """Get a policy version."""
        try:
            policy_version_id = uuid.UUID(policy_version_id_str)
            policy_version = await stats_repo.get_policy_version_with_name(policy_version_id)
            if policy_version is None:
                raise HTTPException(status_code=404, detail=f"Policy version {policy_version_id} not found")
            return policy_version
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get policy version: {str(e)}") from e

    @router.put("/policies/versions/{policy_version_id_str}/tags")
    @timed_route("update_policy_version_tags")
    async def update_policy_version_tags_route(
        policy_version_id_str: str, tags: Annotated[dict[str, str], Body(...)], user: UserOrToken
    ) -> UUIDResponse:
        try:
            policy_version_id = uuid.UUID(policy_version_id_str)
            await stats_repo.upsert_policy_version_tags(policy_version_id, tags)
            return UUIDResponse(id=policy_version_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update policy version tags: {str(e)}") from e

    @router.post("/policies/submit")
    @timed_route("submit_policy")
    async def submit_policy(file: UploadFile, user: UserOrToken, name: str = Form(...)) -> UUIDResponse:
        if not file.filename or not file.filename.endswith(".zip"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a .zip file",
            )

        # Generate unique submission ID
        submission_uuid = uuid.uuid4()

        # Construct S3 path: cogames/submissions/{user_id}/{uuid}.zip
        s3_key = f"cogames/submissions/{user}/{submission_uuid}.zip"

        # Upload file to S3 using streaming to avoid loading entire file into memory
        try:
            # Use SpooledTemporaryFile to keep small files in memory, large ones on disk
            # max_size=100MB - files smaller than this stay in memory for performance
            with tempfile.SpooledTemporaryFile(max_size=100 * 1024 * 1024) as temp_file:
                # Stream file content in chunks to temporary file
                while chunk := await file.read(8192):  # 8KB chunks
                    temp_file.write(chunk)
                temp_file.seek(0)

                # Use async S3 client to upload without blocking event loop
                session = aioboto3.Session()
                async with session.client("s3") as s3_client:  # type: ignore
                    await s3_client.upload_fileobj(
                        temp_file,
                        OBSERVATORY_S3_BUCKET,
                        s3_key,
                        ExtraArgs={"ContentType": "application/zip"},
                    )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file to S3: {str(e)}",
            ) from e

        try:
            return await _create_policy_version_from_s3_key(name=name, user_id=user, s3_key=s3_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to submit policy: {str(e)}") from e

    @router.post("/policies/submit/presigned-url")
    @timed_route("get_submit_policy_presigned_url")
    async def get_submit_policy_presigned_url(user: UserOrToken) -> PresignedUploadUrlResponse:
        """Generate a presigned URL for direct S3 upload of a policy submission zip."""
        try:
            upload_id = uuid.uuid4()
            s3_key = f"cogames/submissions/{user}/{upload_id}.zip"

            from botocore.config import Config

            session = aioboto3.Session()
            async with session.client("s3", config=Config(signature_version="s3v4")) as s3_client:  # type: ignore
                presigned_url = await s3_client.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": OBSERVATORY_S3_BUCKET,
                        "Key": s3_key,
                        "ContentType": "application/zip",
                    },
                    ExpiresIn=3600,  # 1 hour expiration
                )

            return PresignedUploadUrlResponse(upload_url=presigned_url, s3_key=s3_key, upload_id=upload_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}") from e

    @router.post("/policies/submit/complete")
    @timed_route("complete_policy_submit")
    async def complete_policy_submit(request: CompletePolicySubmitRequest, user: UserOrToken) -> UUIDResponse:
        """Finalize a policy submission after the client uploads the zip to S3."""
        s3_key = f"cogames/submissions/{user}/{request.upload_id}.zip"

        try:
            session = aioboto3.Session()
            async with session.client("s3") as s3_client:  # type: ignore
                await s3_client.head_object(Bucket=OBSERVATORY_S3_BUCKET, Key=s3_key)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Uploaded submission not found in S3: {str(e)}") from e

        try:
            return await _create_policy_version_from_s3_key(name=request.name, user_id=user, s3_key=s3_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to submit policy: {str(e)}") from e

    @router.post("/episodes/bulk_upload/presigned-url")
    @timed_route("get_bulk_upload_presigned_url")
    async def get_bulk_upload_presigned_url(user: UserOrToken) -> PresignedUploadUrlResponse:
        """Generate a presigned URL for direct S3 upload of episode stats DuckDB file."""
        try:
            # Generate unique upload ID
            upload_id = uuid.uuid4()
            s3_key = f"episodes/{upload_id}.duckdb"

            # Generate presigned URL (uses IAM role from service account)
            from botocore.config import Config

            session = aioboto3.Session()
            async with session.client("s3", config=Config(signature_version="s3v4")) as s3_client:  # type: ignore
                presigned_url = await s3_client.generate_presigned_url(
                    "put_object",
                    Params={
                        "Bucket": OBSERVATORY_S3_BUCKET,
                        "Key": s3_key,
                        "ContentType": "application/octet-stream",
                    },
                    ExpiresIn=3600,  # 1 hour expiration
                )

            return PresignedUploadUrlResponse(upload_url=presigned_url, s3_key=s3_key, upload_id=upload_id)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}") from e

    @router.post("/episodes/bulk_upload/complete")
    @timed_route("complete_bulk_upload")
    async def complete_bulk_upload(
        request: CompleteBulkUploadRequest,
        user: UserOrToken,
    ) -> BulkEpisodeUploadResponse:
        """Complete the bulk upload by processing the DuckDB file from S3.

        This endpoint is called after the client has uploaded the DuckDB file to S3 using the presigned URL.
        It downloads the file from S3, processes it, and writes aggregated episodes to the database.
        """
        try:
            upload_id = request.upload_id
            s3_key = f"episodes/{upload_id}.duckdb"
            s3_uri = f"s3://{OBSERVATORY_S3_BUCKET}/{s3_key}"

            # Download DuckDB file from S3 to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb") as temp_file:
                temp_file_path = temp_file.name

            session = aioboto3.Session()
            async with session.client("s3") as s3_client:  # type: ignore
                # Download file
                await s3_client.download_file(OBSERVATORY_S3_BUCKET, s3_key, temp_file_path)

            # Read episodes from DuckDB and aggregate
            from metta.app_backend.episode_stats_db import (
                read_agent_metrics,
                read_agent_policies,
                read_episode_tags,
                read_episodes,
            )

            conn = duckdb.connect(temp_file_path, read_only=True)

            episodes = read_episodes(conn)

            episodes_created = 0
            for episode_row in episodes:
                episode_id = uuid.UUID(episode_row[0]) if isinstance(episode_row[0], str) else episode_row[0]
                primary_pv_id = (
                    uuid.UUID(episode_row[1]) if isinstance(episode_row[1], str) and episode_row[1] else None
                )
                replay_url = episode_row[2]
                thumbnail_url = episode_row[3]
                attributes = episode_row[4] or {}
                eval_task_id = uuid.UUID(episode_row[5]) if isinstance(episode_row[5], str) and episode_row[5] else None

                # Get tags for this episode
                tags = read_episode_tags(conn, str(episode_id))

                # Get agent policies for this episode
                agent_policy_map_str = read_agent_policies(conn, str(episode_id))
                agent_policy_map = {agent_id: uuid.UUID(pv_id) for agent_id, pv_id in agent_policy_map_str.items()}

                # Get agent metrics for this episode
                agent_metrics_result = read_agent_metrics(conn, str(episode_id))

                # Aggregate metrics by policy version (sum across agents with same policy)
                policy_metrics: dict[uuid.UUID, dict[str, float]] = {}
                for agent_id, metric_name, metric_value in agent_metrics_result:
                    # Only whitelist "reward" metric for now
                    if metric_name != "reward":
                        continue

                    pv_id = agent_policy_map.get(int(agent_id))
                    if pv_id is None:
                        continue

                    if pv_id not in policy_metrics:
                        policy_metrics[pv_id] = {}

                    if metric_name not in policy_metrics[pv_id]:
                        policy_metrics[pv_id][metric_name] = 0.0

                    policy_metrics[pv_id][metric_name] += float(metric_value)

                # Count agents per policy for the episode_policies table
                policy_agent_counts: dict[uuid.UUID, int] = {}
                for _agent_id, pv_id in agent_policy_map.items():
                    policy_agent_counts[pv_id] = policy_agent_counts.get(pv_id, 0) + 1

                # Prepare data for record_episode
                policy_versions = [(pv_id, count) for pv_id, count in policy_agent_counts.items()]
                policy_metrics_list = [
                    (pv_id, metric_name, value)
                    for pv_id, metrics in policy_metrics.items()
                    for metric_name, value in metrics.items()
                ]

                # Write to database
                await stats_repo.record_episode(
                    id=episode_id,
                    data_uri=s3_uri,
                    primary_pv_id=primary_pv_id,
                    replay_url=replay_url,
                    attributes=attributes,
                    eval_task_id=eval_task_id,
                    thumbnail_url=thumbnail_url,
                    tags=tags,
                    policy_versions=policy_versions,
                    policy_metrics=policy_metrics_list,
                )

                episodes_created += 1

            conn.close()

            return BulkEpisodeUploadResponse(episodes_created=episodes_created, duckdb_s3_uri=s3_uri)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to complete bulk upload: {str(e)}") from e

    @router.get("/policies")
    @timed_route("get_policies")
    async def get_policies(
        name: Optional[str] = None,
        version: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PolicyVersionsResponse:
        try:
            entries, total_count = await stats_repo.get_policy_versions(
                name=name,
                version=version,
                limit=limit,
                offset=offset,
            )
            return PolicyVersionsResponse(entries=entries, total_count=total_count)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get policies: {str(e)}") from e

    @router.get("/policies/my-versions")
    @timed_route("get_my_policy_versions")
    async def get_my_policy_versions(user: UserOrToken) -> MyPolicyVersionsResponse:
        """
        Get all policy versions for the current user.

        This route is used on https://softmax.com/alignmentleague to get the polices for the current user.
        """
        try:
            policy_versions = await stats_repo.get_user_policy_versions(user)
            return MyPolicyVersionsResponse(entries=policy_versions)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get policy versions: {str(e)}") from e

    @router.post("/episodes/query", response_model=EpisodeQueryResponse)
    @timed_route("query_episodes")
    async def query_episodes(request: EpisodeQueryRequest, user: UserOrToken) -> EpisodeQueryResponse:
        """Query episodes by primary policy versions, tags, and replay availability."""
        try:
            episodes = await stats_repo.get_episodes(
                primary_policy_version_ids=request.primary_policy_version_ids,
                episode_ids=request.episode_ids,
                tag_filters=request.tag_filters,
                limit=request.limit,
                offset=request.offset,
            )
            return EpisodeQueryResponse(episodes=episodes)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to query episodes: {str(e)}") from e

    return router
