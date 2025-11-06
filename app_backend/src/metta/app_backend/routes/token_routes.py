import typing

import fastapi
import fastapi.responses
import pydantic

import metta.app_backend.auth
import metta.app_backend.metta_repo
import metta.app_backend.route_logger


# Request/Response Models
class TokenCreate(pydantic.BaseModel):
    name: str


class TokenResponse(pydantic.BaseModel):
    token: str


class TokenInfo(pydantic.BaseModel):
    id: str
    name: str
    created_at: str
    expiration_time: str
    last_used_at: str | None


class TokenListResponse(pydantic.BaseModel):
    tokens: typing.List[TokenInfo]


def create_token_router(metta_repo: metta.app_backend.metta_repo.MettaRepo) -> fastapi.APIRouter:
    """Create a token management router with the given MettaRepo instance."""
    router = fastapi.APIRouter(prefix="/tokens", tags=["tokens"])

    @router.post("", response_model=TokenResponse)
    @metta.app_backend.route_logger.timed_route("create_token")
    async def create_token(token_data: TokenCreate, user_email: metta.app_backend.auth.UserEmail) -> TokenResponse:
        """Create a new machine token for the authenticated user."""
        try:
            token = await metta_repo.create_machine_token(user_email, token_data.name)
            return TokenResponse(token=token)
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to create token: {str(e)}") from e

    @router.get("/cli")
    @metta.app_backend.route_logger.timed_route("create_cli_token")
    async def create_cli_token(
        user_email: metta.app_backend.auth.UserEmail,
        callback: str = fastapi.Query(..., description="Callback URL to redirect to with token"),
    ) -> fastapi.responses.RedirectResponse:
        """Create a machine token and redirect to callback URL with token parameter."""
        try:
            # Validate the callback URL
            if not callback.startswith("http://127.0.0.1"):
                raise fastapi.HTTPException(status_code=400, detail="Invalid callback URL")

            # Create the machine token
            token = await metta_repo.create_machine_token(user_email, name="CLI Token")

            # Build the redirect URL with token parameter
            import urllib.parse

            redirect_url = f"{callback}?{urllib.parse.urlencode({'token': token})}"
            return fastapi.responses.RedirectResponse(url=redirect_url)

        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to create CLI token: {str(e)}") from e

    @router.get("", response_model=TokenListResponse)
    @metta.app_backend.route_logger.timed_route("list_tokens")
    async def list_tokens(user_email: metta.app_backend.auth.UserEmail) -> TokenListResponse:
        """List all machine tokens for the authenticated user."""
        try:
            token_rows = await metta_repo.list_machine_tokens(user_email)
            tokens = [
                TokenInfo(
                    id=str(token_row.id),
                    name=token_row.name,
                    created_at=str(token_row.created_at),
                    expiration_time=str(token_row.expiration_time),
                    last_used_at=str(token_row.last_used_at) if token_row.last_used_at else None,
                )
                for token_row in token_rows
            ]
            return TokenListResponse(tokens=tokens)
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to list tokens: {str(e)}") from e

    @router.delete("/{token_id}")
    @metta.app_backend.route_logger.timed_route("delete_token")
    async def delete_token(token_id: str, user_email: metta.app_backend.auth.UserEmail) -> dict[str, str]:
        """Delete a machine token for the authenticated user."""
        try:
            success = await metta_repo.delete_machine_token(user_email, token_id)
            if not success:
                raise fastapi.HTTPException(status_code=404, detail="Token not found")
            return {"message": "Token deleted successfully"}
        except fastapi.HTTPException:
            raise
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to delete token: {str(e)}") from e

    return router
