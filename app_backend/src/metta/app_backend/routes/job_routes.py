import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import col, select

from metta.app_backend.auth import UserOrToken
from metta.app_backend.database import get_session
from metta.app_backend.job_runner.dispatcher import dispatch_job
from metta.app_backend.models.job_request import JobRequest, JobRequestCreate, JobRequestUpdate, JobStatus, JobType
from metta.app_backend.route_logger import timed_http_handler

logger = logging.getLogger(__name__)

VALID_TRANSITIONS = {
    JobStatus.pending: {JobStatus.dispatched},
    JobStatus.dispatched: {JobStatus.running, JobStatus.failed},
    JobStatus.running: {JobStatus.completed, JobStatus.failed},
}


def create_job_router() -> APIRouter:
    router = APIRouter(prefix="/jobs", tags=["jobs"])

    @router.post("/batch")
    @timed_http_handler
    async def create_jobs_batch(jobs: list[JobRequestCreate], user: UserOrToken) -> list[UUID]:
        if not jobs:
            return []

        # Create all jobs in db as pending
        db_jobs = []
        async with get_session() as session:
            for job_create in jobs:
                db_job = JobRequest(**job_create.model_dump(), user_id=user)
                session.add(db_job)
                db_jobs.append(db_job)
            await session.commit()
            # Capture IDs before session closes
            job_data = [(j.id, j) for j in db_jobs]

        class _DispatchResult(BaseModel):
            k8s_job_name: str | None = None
            error: str | None = None
            time: datetime

        # Dispatch each job (outside DB session)
        dispatch_results: dict[UUID, _DispatchResult] = {}
        for job_id, db_job in job_data:
            try:
                dispatch_results[job_id] = _DispatchResult(k8s_job_name=dispatch_job(db_job), time=datetime.now(UTC))
            except Exception as e:
                logger.error(f"Failed to dispatch job {job_id}: {e}", exc_info=True)
                dispatch_results[job_id] = _DispatchResult(error=str(e), time=datetime.now(UTC))

        # Update DB with dispatch results
        async with get_session() as session:
            query = await session.execute(select(JobRequest).where(col(JobRequest.id).in_(dispatch_results.keys())))
            job_requests = list(query.scalars().all())
            for job_request in job_requests:
                result = dispatch_results.get(job_request.id)
                if not result:
                    logger.error(f"Job {job_request.id} not found in dispatch results")
                    continue
                if result.k8s_job_name:
                    job_request.status = JobStatus.dispatched
                    job_request.worker = result.k8s_job_name
                    job_request.dispatched_at = result.time
                else:
                    job_request.status = JobStatus.failed
                    job_request.error = result.error
            await session.commit()
            return [job_request.id for job_request in job_requests]

    @router.get("")
    @timed_http_handler
    async def list_jobs(
        _user: UserOrToken,
        job_type: JobType | None = Query(default=None),
        statuses: list[JobStatus] | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> list[JobRequest]:
        async with get_session() as session:
            query = select(JobRequest).order_by(col(JobRequest.created_at).desc()).offset(offset).limit(limit)
            if statuses:
                query = query.where(col(JobRequest.status).in_(statuses))
            if job_type:
                query = query.where(col(JobRequest.job_type) == job_type)
            result = await session.execute(query)
            return list(result.scalars().all())

    @router.get("/{job_id}")
    @timed_http_handler
    async def get_job(job_id: UUID, _user: UserOrToken) -> JobRequest:
        async with get_session() as session:
            result = await session.execute(select(JobRequest).where(JobRequest.id == job_id))
            row = result.scalar_one_or_none()
            if not row:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            return row

    @router.post("/{job_id}")
    @timed_http_handler
    async def update_job(job_id: UUID, request: JobRequestUpdate, _user: UserOrToken) -> JobRequest:
        async with get_session() as session:
            result = await session.execute(select(JobRequest).where(JobRequest.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            if request.status is not None:
                allowed = VALID_TRANSITIONS.get(job.status, set())
                if request.status not in allowed:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Cannot transition from {job.status} to {request.status}",
                    )

                job.status = request.status

                if request.status == JobStatus.running:
                    job.running_at = datetime.now(UTC)
                    if request.worker:
                        job.worker = request.worker

            if request.error is not None:
                job.error = request.error

            if request.result is not None:
                job.result = request.result
                job.completed_at = datetime.now(UTC)

            await session.commit()
            await session.refresh(job)
            return job

    return router
