from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import col, select

from metta.app_backend.auth import UserOrToken
from metta.app_backend.models.database import get_session
from metta.app_backend.models.job_request import JobRequest, JobStatus, JobType
from metta.app_backend.route_logger import timed_http_handler


def create_job_router(job_type: JobType, prefix: str) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=[f"{job_type.value}_jobs"])

    @router.post("/batch")
    @timed_http_handler
    async def create_jobs_batch(jobs: list[dict[str, Any]], user: UserOrToken) -> list[UUID]:
        if not jobs:
            return []

        db_jobs = [JobRequest(job_type=job_type, job=j) for j in jobs]
        async with get_session() as session:
            session.add_all(db_jobs)
            await session.commit()
            return [j.id for j in db_jobs]

    @router.get("")
    @timed_http_handler
    async def list_jobs(
        user: UserOrToken,
        status: JobStatus | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> list[JobRequest]:
        async with get_session() as session:
            query = (
                select(JobRequest)
                .where(JobRequest.job_type == job_type)
                .order_by(col(JobRequest.created_at).desc())
                .offset(offset)
                .limit(limit)
            )
            if status:
                query = query.where(JobRequest.status == status)
            result = await session.execute(query)
            rows = result.scalars().all()
            return list(rows)

    @router.get("/{job_id}")
    @timed_http_handler
    async def get_job(job_id: UUID, user: UserOrToken) -> JobRequest:
        async with get_session() as session:
            result = await session.execute(
                select(JobRequest).where(JobRequest.id == job_id, JobRequest.job_type == job_type)
            )
            row = result.scalar_one_or_none()
            if not row:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            return row

    return router
