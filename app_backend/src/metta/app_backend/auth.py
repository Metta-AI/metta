from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Request, status

from metta.app_backend.config import settings


def get_user_from_header(request: Request) -> str | None:
    if settings.DEBUG_USER_EMAIL:
        return settings.DEBUG_USER_EMAIL

    user_id = request.headers.get("X-User-Id")
    auth_secret = request.headers.get("X-Auth-Secret")

    if user_id and auth_secret and auth_secret == settings.OBSERVATORY_AUTH_SECRET:
        return user_id
    return None


async def get_user_from_token(request: Request) -> str | None:
    token = request.headers.get("X-Auth-Token")
    if token:
        return await validate_token_via_login_service(token)
    return None


async def user_from_header_or_token(request: Request) -> str | None:
    user_id = get_user_from_header(request)
    if user_id:
        return user_id

    return await get_user_from_token(request)


async def user_from_header_or_token_or_raise(request: Request) -> str:
    user_id = await user_from_header_or_token(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to authenticate",
        )
    return user_id


# Dependency types for use in route decorators
UserOrToken = Annotated[str, Depends(user_from_header_or_token_or_raise)]
OptionalUserOrToken = Annotated[str | None, Depends(user_from_header_or_token)]


async def validate_token_via_login_service(token: str) -> str | None:
    """Validate a machine token via the login service and return the user_id if valid."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.LOGIN_SERVICE_URL}/api/validate",
                headers={"X-Auth-Token": token},
                timeout=5.0,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("valid"):
                    user_info = data.get("user", {})
                    return user_info.get("id")
            return None
    except Exception:
        return None
