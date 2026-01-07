"""
Observatory backend treats users as opaque user ids. The ids are managed by softmax.com login service.

There are several ways to authenticate requests:

1) `X-User-Id` and `X-Auth-Secret` header pair, where `X-Auth-Secret` is the secret static key that is shared between
the backend and softmax.com login service. This is used for requests from softmax.com.

2) `X-Auth-Token` header
"""

from typing import Annotated, Optional

import httpx
from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from metta.app_backend.config import settings


class User(BaseModel):
    id: str
    email: str
    is_softmax_team_member: bool = Field(default=False, alias="is_softmax_team_member")


def get_user_from_header(request: Request) -> Optional[User]:
    if settings.DEBUG_USER_EMAIL:
        return User(id="debug_user_id", email=settings.DEBUG_USER_EMAIL, is_softmax_team_member=True)

    if not settings.OBSERVATORY_AUTH_SECRET:
        return None

    # check static secret key that allows softmax.com to bypass token validation
    auth_secret = request.headers.get("X-Auth-Secret")
    if auth_secret != settings.OBSERVATORY_AUTH_SECRET:
        return None

    user_id = request.headers.get("X-User-Id")
    user_email = request.headers.get(
        "X-User-Email",
        "",  # TODO - allowed empty value for now - current softmax.com doesn't set this header
    )
    is_softmax_team_member = request.headers.get("X-User-Is-Softmax-Team-Member", "false") == "true"

    if user_id:
        return User(id=user_id, email=user_email, is_softmax_team_member=is_softmax_team_member)
    return None


async def get_user_from_token(request: Request) -> Optional[User]:
    token = request.headers.get("X-Auth-Token")
    if token:
        return await validate_token_via_login_service(token)
    return None


async def get_user(request: Request) -> Optional[User]:
    user = get_user_from_header(request)
    if user:
        return user

    return await get_user_from_token(request)


async def get_user_or_raise(request: Request) -> User:
    user = await get_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to authenticate",
        )
    return user


async def get_softmax_user_or_raise(request: Request) -> User:
    user = await get_user_or_raise(request)
    if not user.is_softmax_team_member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not a softmax team member",
        )
    return user


# Dependency types for use in route decorators
CheckUser = Annotated[User, Depends(get_user_or_raise)]
CheckSoftmaxUser = Annotated[User, Depends(get_softmax_user_or_raise)]
CheckMaybeUser = Annotated[Optional[User], Depends(get_user)]


async def validate_token_via_login_service(token: str) -> Optional[User]:
    """Validate a machine token via the login service and return the user if valid."""
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
                    return User(
                        id=user_info.get("id"),
                        email=user_info.get("email"),
                        is_softmax_team_member=user_info.get("isSoftmaxTeamMember", False),
                    )
            return None
    except Exception:
        return None
