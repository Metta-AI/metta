import tempfile
import uuid

import aioboto3
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from metta.app_backend.auth import validate_token_via_login_service
from metta.app_backend.metta_repo import MettaRepo
from metta.common.util.constants import SOFTMAX_S3_BUCKET


class SubmitPolicyResponse(BaseModel):
    message: str
    submission_id: str


def create_cogames_router(stats_repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/cogames", tags=["cogames"])

    @router.post("/submit_policy", response_model=SubmitPolicyResponse)
    async def submit_policy(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        name: str | None = Form(None),  # noqa: B008
    ) -> SubmitPolicyResponse:
        """Submit a policy zip file for CoGames.

        Authenticates via machine token from the login service,
        uploads the file to S3 using streaming, and records the submission in the database.
        """
        # Extract machine token from request headers
        token = request.headers.get("X-Auth-Token")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="X-Auth-Token header required",
            )

        # Validate token via login service
        user_id = await validate_token_via_login_service(token)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )

        # Validate file is a zip
        if not file.filename or not file.filename.endswith(".zip"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a .zip file",
            )

        # Generate unique submission ID
        submission_uuid = uuid.uuid4()

        # Construct S3 path: cogames/submissions/{user_id}/{uuid}.zip
        s3_key = f"cogames/submissions/{user_id}/{submission_uuid}.zip"
        s3_path = f"s3://{SOFTMAX_S3_BUCKET}/{s3_key}"

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
                async with session.client("s3") as s3_client:
                    await s3_client.upload_fileobj(
                        temp_file,
                        SOFTMAX_S3_BUCKET,
                        s3_key,
                        ExtraArgs={"ContentType": "application/zip"},
                    )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file to S3: {str(e)}",
            ) from e

        # Record submission in database
        try:
            submission_id = await stats_repo.create_cogames_submission(
                submission_id=submission_uuid,
                user_id=user_id,
                s3_path=s3_path,
                name=name,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to record submission in database: {str(e)}",
            ) from e

        return SubmitPolicyResponse(
            message="Policy submitted successfully",
            submission_id=str(submission_id),
        )

    return router
