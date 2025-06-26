from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app_backend.auth import UserEmail
from app_backend.metta_repo import MettaRepo


# Request/Response Models
class TokenCreate(BaseModel):
    name: str


class TokenResponse(BaseModel):
    token: str


class TokenInfo(BaseModel):
    id: str
    name: str
    created_at: str
    expiration_time: str
    last_used_at: str | None


class TokenListResponse(BaseModel):
    tokens: List[TokenInfo]


def create_token_router(metta_repo: MettaRepo) -> APIRouter:
    """Create a token management router with the given MettaRepo instance."""
    router = APIRouter(prefix="/tokens", tags=["tokens"])

    @router.post("", response_model=TokenResponse)
    async def create_token(token_data: TokenCreate, user_email: UserEmail) -> TokenResponse:
        """Create a new machine token for the authenticated user."""
        try:
            token = metta_repo.create_machine_token(user_email, token_data.name)
            return TokenResponse(token=token)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create token: {str(e)}") from e

    @router.get("/cli")
    async def create_cli_token(
        user_email: UserEmail, callback: str = Query(..., description="Callback URL to redirect to with token")
    ) -> RedirectResponse:
        """Create a machine token and redirect to callback URL with token parameter."""
        try:
            # Validate the callback URL
            if not callback.startswith("http://127.0.0.1"):
                raise HTTPException(status_code=400, detail="Invalid callback URL")

            # Create the machine token
            token = metta_repo.create_machine_token(user_email, name="CLI Token")

            # Build the redirect URL with token parameter
            from urllib.parse import urlencode

            redirect_url = f"{callback}?{urlencode({'token': token})}"
            return RedirectResponse(url=redirect_url)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create CLI token: {str(e)}") from e

    @router.get("", response_model=TokenListResponse)
    async def list_tokens(user_email: UserEmail) -> TokenListResponse:
        """List all machine tokens for the authenticated user."""
        try:
            token_dicts = metta_repo.list_machine_tokens(user_email)
            tokens = [
                TokenInfo(
                    id=token_dict["id"],
                    name=token_dict["name"],
                    created_at=str(token_dict["created_at"]),
                    expiration_time=str(token_dict["expiration_time"]),
                    last_used_at=str(token_dict["last_used_at"]) if token_dict["last_used_at"] else None,
                )
                for token_dict in token_dicts
            ]
            return TokenListResponse(tokens=tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list tokens: {str(e)}") from e

    @router.delete("/{token_id}")
    async def delete_token(token_id: str, user_email: UserEmail) -> dict[str, str]:
        """Delete a machine token for the authenticated user."""
        try:
            success = metta_repo.delete_machine_token(user_email, token_id)
            if not success:
                raise HTTPException(status_code=404, detail="Token not found")
            return {"message": "Token deleted successfully"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete token: {str(e)}") from e

    return router
