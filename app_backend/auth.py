from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Request, status

from app_backend.metta_repo import MettaRepo


def get_user_email(request: Request) -> str:
    """Extract user email from request headers."""
    user_email = request.headers.get("X-Auth-Request-Email")
    if not user_email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="X-Auth-Request-Email header required")
    return user_email


def create_user_or_token_dependency(metta_repo: MettaRepo) -> Callable[[Request], str]:
    """Create a dependency function that validates either user email or machine token."""

    def get_user_or_token_user(request: Request) -> str:
        # First try to get user email from header
        user_email = request.headers.get("X-Auth-Request-Email")
        if user_email:
            return user_email

        # If no email, try to validate token
        token = request.headers.get("X-Auth-Token")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Either X-Auth-Request-Email or X-Auth-Token header required",
            )

        # Validate token and get user_id
        user_id = metta_repo.validate_machine_token(token)
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token")

        return user_id

    return get_user_or_token_user


# Dependency types for use in route decorators
UserEmail = Annotated[str, Depends(get_user_email)]
