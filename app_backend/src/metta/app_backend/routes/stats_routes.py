import tempfile
import uuid
from typing import Annotated, Any, Optional

import aioboto3
import duckdb
from fastapi import APIRouter, Body, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import selectinload
from sqlmodel import col, select

from metta.app_backend.auth import UserOrToken
from metta.app_backend.database import DbSession
from metta.app_backend.metta_repo import (
    EpisodeWithTags,
    MettaRepo,
    PolicyRow,
    PolicyVersionWithName,
    PublicPolicyVersionRow,
)
from metta.app_backend.models.policies import Policy, PolicyVersion
from metta.app_backend.route_logger import timed_http_handler

OBSERVATORY_S3_BUCKET = "observatory-private"


# Request/Response Models
class UUIDResponse(BaseModel):
    id: uuid.UUID


class PolicyVersionResponse(BaseModel):
    id: uuid.UUID
    name: str
    version: int


class PolicyCreate(BaseModel):
    name: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    is_system_policy: bool = False


class PolicyVersionCreate(BaseModel):
    policy_spec: dict[str, Any] = Field(default_factory=dict)
    git_hash: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    s3_path: str | None = None


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


class PoliciesResponse(BaseModel):
    entries: list[PolicyRow]
    total_count: int


class PolicyVersionsResponse(BaseModel):
    entries: list[PublicPolicyVersionRow]
    total_count: int


def create_stats_router(stats_repo: MettaRepo) -> APIRouter:
    """Create a stats router with the given StatsRepo instance."""
    router = APIRouter(prefix="/stats", tags=["stats"])

    async def _create_policy_version_from_s3_key(name: str, user_id: str, s3_key: str) -> PolicyVersionResponse:
        s3_path = f"s3://{OBSERVATORY_S3_BUCKET}/{s3_key}"
        policy_id = await stats_repo.upsert_policy(name=name, user_id=user_id, attributes={})
        policy_version_id = await stats_repo.create_policy_version(
            policy_id=policy_id,
            s3_path=s3_path,
            git_hash=None,
            policy_spec={},
            attributes={},
        )
        pv = await stats_repo.get_policy_version_with_name(policy_version_id)
        if pv is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve created policy version")
        return PolicyVersionResponse(id=policy_version_id, name=pv.name, version=pv.version)

    @router.post("/policies")
    @timed_http_handler
    async def upsert_policy(policy: PolicyCreate, user: UserOrToken) -> UUIDResponse:
        if policy.is_system_policy:
            user_id = "system"
        else:
            user_id = user

        policy_id = await stats_repo.upsert_policy(name=policy.name, user_id=user_id, attributes=policy.attributes)
        return UUIDResponse(id=policy_id)

    @router.post("/policies/{policy_id_str}/versions")
    @timed_http_handler
    async def create_policy_version(
        policy_id_str: str, policy_version: PolicyVersionCreate, user: UserOrToken
    ) -> UUIDResponse:
        policy_id = uuid.UUID(policy_id_str)
        policy_version_id = await stats_repo.create_policy_version(
            policy_id=policy_id,
            s3_path=policy_version.s3_path,
            git_hash=policy_version.git_hash,
            policy_spec=policy_version.policy_spec,
            attributes=policy_version.attributes,
        )
        return UUIDResponse(id=policy_version_id)

    @router.get("/policies/versions/{policy_version_id_str}")
    @timed_http_handler
    async def get_policy_version(policy_version_id_str: str) -> PolicyVersionWithName:
        policy_version_id = uuid.UUID(policy_version_id_str)
        policy_version = await stats_repo.get_policy_version_with_name(policy_version_id)
        if policy_version is None:
            raise HTTPException(status_code=404, detail=f"Policy version {policy_version_id} not found")
        return policy_version

    @router.get("/policies/{policy_id}")
    @timed_http_handler
    async def get_policy_by_id(policy_id: str, user: UserOrToken) -> PublicPolicyVersionRow:
        """Get a single policy version by ID.

        Note: Despite the parameter name 'policy_id', this endpoint expects
        a policy_version_id (UUID). This naming matches the frontend's convention.

        The frontend page at /alignmentleague/policy/[id] relies on this endpoint.
        """
        try:
            policy_version_id = uuid.UUID(policy_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid UUID format: {policy_id}") from None

        policy_version = await stats_repo.get_public_policy_version_by_id(policy_version_id)

        if policy_version is None:
            raise HTTPException(status_code=404, detail=f"Policy version {policy_id} not found")

        return policy_version

    @router.put("/policies/versions/{policy_version_id_str}/tags")
    @timed_http_handler
    async def update_policy_version_tags_route(
        policy_version_id_str: str, tags: Annotated[dict[str, str], Body(...)], user: UserOrToken
    ) -> UUIDResponse:
        policy_version_id = uuid.UUID(policy_version_id_str)
        await stats_repo.upsert_policy_version_tags(policy_version_id, tags)
        return UUIDResponse(id=policy_version_id)

    @router.post("/policies/submit")
    @timed_http_handler
    async def submit_policy(file: UploadFile, user: UserOrToken, name: str = Form(...)) -> PolicyVersionResponse:
        if not file.filename or not file.filename.endswith(".zip"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a .zip file",
            )

        submission_uuid = uuid.uuid4()
        s3_key = f"cogames/submissions/{user}/{submission_uuid}.zip"

        with tempfile.SpooledTemporaryFile(max_size=100 * 1024 * 1024) as temp_file:
            while chunk := await file.read(8192):
                temp_file.write(chunk)
            temp_file.seek(0)

            session = aioboto3.Session()
            async with session.client("s3") as s3_client:  # type: ignore
                await s3_client.upload_fileobj(
                    temp_file,
                    OBSERVATORY_S3_BUCKET,
                    s3_key,
                    ExtraArgs={"ContentType": "application/zip"},
                )

        return await _create_policy_version_from_s3_key(name=name, user_id=user, s3_key=s3_key)

    @router.post("/policies/submit/presigned-url")
    @timed_http_handler
    async def get_submit_policy_presigned_url(user: UserOrToken) -> PresignedUploadUrlResponse:
        from botocore.config import Config

        upload_id = uuid.uuid4()
        s3_key = f"cogames/submissions/{user}/{upload_id}.zip"

        session = aioboto3.Session()
        async with session.client("s3", config=Config(signature_version="s3v4")) as s3_client:  # type: ignore
            presigned_url = await s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": OBSERVATORY_S3_BUCKET,
                    "Key": s3_key,
                    "ContentType": "application/zip",
                },
                ExpiresIn=3600,
            )

        return PresignedUploadUrlResponse(upload_url=presigned_url, s3_key=s3_key, upload_id=upload_id)

    @router.post("/policies/submit/complete")
    @timed_http_handler
    async def complete_policy_submit(request: CompletePolicySubmitRequest, user: UserOrToken) -> PolicyVersionResponse:
        s3_key = f"cogames/submissions/{user}/{request.upload_id}.zip"

        try:
            session = aioboto3.Session()
            async with session.client("s3") as s3_client:  # type: ignore
                await s3_client.head_object(Bucket=OBSERVATORY_S3_BUCKET, Key=s3_key)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Uploaded submission not found in S3: {str(e)}") from e

        return await _create_policy_version_from_s3_key(name=request.name, user_id=user, s3_key=s3_key)

    @router.post("/episodes/bulk_upload/presigned-url")
    @timed_http_handler
    async def get_bulk_upload_presigned_url(user: UserOrToken) -> PresignedUploadUrlResponse:
        from botocore.config import Config

        upload_id = uuid.uuid4()
        s3_key = f"episodes/{upload_id}.duckdb"

        session = aioboto3.Session()
        async with session.client("s3", config=Config(signature_version="s3v4")) as s3_client:  # type: ignore
            presigned_url = await s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": OBSERVATORY_S3_BUCKET,
                    "Key": s3_key,
                    "ContentType": "application/octet-stream",
                },
                ExpiresIn=3600,
            )

        return PresignedUploadUrlResponse(upload_url=presigned_url, s3_key=s3_key, upload_id=upload_id)

    @router.post("/episodes/bulk_upload/complete")
    @timed_http_handler
    async def complete_bulk_upload(
        request: CompleteBulkUploadRequest,
        user: UserOrToken,
    ) -> BulkEpisodeUploadResponse:
        from metta.app_backend.episode_stats_db import (
            read_agent_metrics,
            read_agent_policies,
            read_episode_tags,
            read_episodes,
        )

        upload_id = request.upload_id
        s3_key = f"episodes/{upload_id}.duckdb"
        s3_uri = f"s3://{OBSERVATORY_S3_BUCKET}/{s3_key}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb") as temp_file:
            temp_file_path = temp_file.name

        session = aioboto3.Session()
        async with session.client("s3") as s3_client:  # type: ignore
            await s3_client.download_file(OBSERVATORY_S3_BUCKET, s3_key, temp_file_path)

        conn = duckdb.connect(temp_file_path, read_only=True)

        episodes = read_episodes(conn)

        episodes_created = 0
        for episode_row in episodes:
            episode_id = uuid.UUID(episode_row[0]) if isinstance(episode_row[0], str) else episode_row[0]
            primary_pv_id = uuid.UUID(episode_row[1]) if isinstance(episode_row[1], str) and episode_row[1] else None
            replay_url = episode_row[2]
            thumbnail_url = episode_row[3]
            attributes = episode_row[4] or {}
            eval_task_id = uuid.UUID(episode_row[5]) if isinstance(episode_row[5], str) and episode_row[5] else None

            tags = read_episode_tags(conn, str(episode_id))

            agent_policy_map_str = read_agent_policies(conn, str(episode_id))
            agent_policy_map = {agent_id: uuid.UUID(pv_id) for agent_id, pv_id in agent_policy_map_str.items()}

            agent_metrics_result = read_agent_metrics(conn, str(episode_id))

            policy_metrics: dict[uuid.UUID, dict[str, float]] = {}
            for agent_id, metric_name, metric_value in agent_metrics_result:
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

            policy_agent_counts: dict[uuid.UUID, int] = {}
            for _agent_id, pv_id in agent_policy_map.items():
                policy_agent_counts[pv_id] = policy_agent_counts.get(pv_id, 0) + 1

            policy_versions = [(pv_id, count) for pv_id, count in policy_agent_counts.items()]
            policy_metrics_list = [
                (pv_id, metric_name, value)
                for pv_id, metrics in policy_metrics.items()
                for metric_name, value in metrics.items()
            ]

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

    @router.get("/policies")
    @timed_http_handler
    async def get_policies(
        name_exact: Optional[str] = None,
        name_fuzzy: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PoliciesResponse:
        entries, total_count = await stats_repo.get_policies(
            name_exact=name_exact,
            name_fuzzy=name_fuzzy,
            limit=limit,
            offset=offset,
        )
        return PoliciesResponse(entries=entries, total_count=total_count)

    @router.get("/policy-versions")
    @timed_http_handler
    async def get_policy_versions(
        name_exact: Optional[str] = None,
        name_fuzzy: Optional[str] = None,
        version: Optional[int] = None,
        policy_version_ids: Optional[list[str]] = Query(default=None),
        limit: int = 50,
        offset: int = 0,
    ) -> PolicyVersionsResponse:
        pv_uuids = [uuid.UUID(pv_id) for pv_id in policy_version_ids] if policy_version_ids else None
        entries, total_count = await stats_repo.get_policy_versions(
            name_exact=name_exact,
            name_fuzzy=name_fuzzy,
            version=version,
            policy_version_ids=pv_uuids,
            limit=limit,
            offset=offset,
        )
        return PolicyVersionsResponse(entries=entries, total_count=total_count)

    @router.get("/policies/{policy_id}/versions")
    @timed_http_handler
    async def get_versions_for_policy(
        policy_id: str,
        limit: int = 500,
        offset: int = 0,
    ) -> PolicyVersionsResponse:
        entries, total_count = await stats_repo.get_versions_for_policy(
            policy_id=policy_id,
            limit=limit,
            offset=offset,
        )
        return PolicyVersionsResponse(entries=entries, total_count=total_count)

    @router.get("/policies/my-versions")
    @timed_http_handler
    async def get_my_policy_versions(user: UserOrToken) -> MyPolicyVersionsResponse:
        policy_versions = await stats_repo.get_user_policy_versions(user)
        return MyPolicyVersionsResponse(entries=policy_versions)

    @router.get("/policies/my-versions/lookup")
    @timed_http_handler
    async def lookup_my_policy_version(
        user: UserOrToken,
        session: DbSession,
        name: str = Query(..., description="Policy name"),
        version: Optional[int] = Query(None, description="Version number (latest if omitted)"),
    ) -> PublicPolicyVersionRow:
        query = (
            select(PolicyVersion)
            .join(PolicyVersion.policy)
            .where(Policy.user_id == user, Policy.name == name)
            .options(selectinload(PolicyVersion.policy))
        )
        if version is not None:
            query = query.where(PolicyVersion.version == version)
        else:
            query = query.order_by(col(PolicyVersion.version).desc()).limit(1)

        result = (await session.execute(query)).scalar_one_or_none()
        if result is None:
            version_str = f" v{version}" if version else ""
            raise HTTPException(status_code=404, detail=f"Policy '{name}'{version_str} not found")

        return PublicPolicyVersionRow(
            id=result.id,
            policy_id=result.policy_id,
            created_at=result.created_at,
            policy_created_at=result.policy.created_at,
            user_id=result.policy.user_id,
            name=result.policy.name,
            version=result.version,
        )

    @router.post("/episodes/query", response_model=EpisodeQueryResponse)
    @timed_http_handler
    async def query_episodes(request: EpisodeQueryRequest, user: UserOrToken) -> EpisodeQueryResponse:
        episodes = await stats_repo.get_episodes(
            primary_policy_version_ids=request.primary_policy_version_ids,
            episode_ids=request.episode_ids,
            tag_filters=request.tag_filters,
            limit=request.limit,
            offset=request.offset,
        )
        return EpisodeQueryResponse(episodes=episodes)

    return router
