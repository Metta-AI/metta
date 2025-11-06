import typing

import fastapi
import httpx

import metta.app_backend
import metta.app_backend.metta_repo


def user_from_header(request: fastapi.Request) -> str | None:
    """Extract user email from request headers."""
    return metta.app_backend.config.debug_user_email or request.headers.get("X-Auth-Request-Email")


async def user_from_header_or_token(
    request: fastapi.Request, metta_repo: metta.app_backend.metta_repo.MettaRepo
) -> str | None:
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


async def user_from_header_or_token_or_raise(
    request: fastapi.Request, metta_repo: metta.app_backend.metta_repo.MettaRepo
) -> str:
    user_id = await user_from_header_or_token(request, metta_repo)
    if not user_id:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
            detail="Either X-Auth-Request-Email or X-Auth-Token header required",
        )
    return user_id


def user_from_email_or_raise(request: fastapi.Request) -> str:
    user_email = user_from_header(request)
    if not user_email:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_401_UNAUTHORIZED, detail="X-Auth-Request-Email header required"
        )
    return user_email


def create_user_or_token_dependency(
    metta_repo: metta.app_backend.metta_repo.MettaRepo,
) -> typing.Callable[[fastapi.Request], str]:
    """Create a dependency function that validates either user email or machine token."""

    async def get_user_or_token_user(request: fastapi.Request) -> str:
        return await user_from_header_or_token_or_raise(request, metta_repo)

    return get_user_or_token_user


# Dependency types for use in route decorators
UserEmail = typing.Annotated[str, fastapi.Depends(user_from_email_or_raise)]


async def validate_token_via_login_service(token: str) -> str | None:
    """Validate a machine token via the login service and return the user_id if valid."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{metta.app_backend.config.login_service_url}/api/validate",
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
