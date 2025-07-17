from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Request, status

from metta.app_backend import config
from metta.app_backend.metta_repo import MettaRepo


def user_from_header(request: Request) -> str | None:
    """Extract user email from request headers."""
    return config.debug_user_email or request.headers.get("X-Auth-Request-Email")


async def user_from_header_or_token(request: Request, metta_repo: MettaRepo) -> str | None:
    user_email = user_from_header(request)
    if user_email:
        return user_email

    token = request.headers.get("X-Auth-Token")
    if not token:
        return None

    user_id = await metta_repo.validate_machine_token(token)
    if not user_id:
        return None

    return user_id


async def user_from_header_or_token_or_raise(request: Request, metta_repo: MettaRepo) -> str:
    user_id = await user_from_header_or_token(request, metta_repo)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Either X-Auth-Request-Email or X-Auth-Token header required",
        )
    return user_id


def user_from_email_or_raise(request: Request) -> str:
    user_email = user_from_header(request)
    if not user_email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-Auth-Request-Email header required")
    return user_email


def create_user_or_token_dependency(metta_repo: MettaRepo) -> Callable[[Request], str]:
    """Create a dependency function that validates either user email or machine token."""

    async def get_user_or_token_user(request: Request) -> str:
        return await user_from_header_or_token_or_raise(request, metta_repo)

    return get_user_or_token_user


# Dependency types for use in route decorators
UserEmail = Annotated[str, Depends(user_from_email_or_raise)]
