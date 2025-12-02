from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Request, status

from metta.app_backend import config


async def user_from_header_or_token(request: Request) -> str | None:
    user_email = config.debug_user_email
    if user_email:
        return user_email

    token = request.headers.get("X-Auth-Token")
    if not token:
        return None

    user_id = await validate_token_via_login_service(token)
    if not user_id:
        return None

    return user_id


async def user_from_header_or_token_or_raise(request: Request) -> str:
    user_id = await user_from_header_or_token(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Either X-Auth-Request-Email or X-Auth-Token header required",
        )
    return user_id


# Dependency types for use in route decorators
UserOrToken = Annotated[str, Depends(user_from_header_or_token_or_raise)]


async def validate_token_via_login_service(token: str) -> str | None:
    """Validate a machine token via the login service and return the user_id if valid."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{config.login_service_url}/api/validate",
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
