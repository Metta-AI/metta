"""Shared authentication utilities for Metta backend services."""

from typing import Optional
from fastapi import Depends, HTTPException, Header
from pydantic import BaseModel


class User(BaseModel):
    """Basic user model for authentication."""
    username: str
    token: Optional[str] = None


def create_user_or_token_dependency(
    optional: bool = False,
    api_key_header_name: str = "X-API-Key"
) -> callable:
    """Create a dependency for user/token authentication.
    
    Args:
        optional: If True, authentication is optional
        api_key_header_name: Name of the API key header
        
    Returns:
        A FastAPI dependency function
    """
    async def get_current_user(
        authorization: Optional[str] = Header(None),
        api_key: Optional[str] = Header(None, alias=api_key_header_name)
    ) -> Optional[User]:
        """Extract user from authorization header or API key."""
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            return User(username="bearer_user", token=token)
        elif api_key:
            return User(username="api_key_user", token=api_key)
        elif not optional:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated"
            )
        return None
    
    return Depends(get_current_user)


def authenticate(token: str) -> bool:
    """Basic authentication check.
    
    Args:
        token: Authentication token to verify
        
    Returns:
        True if authenticated, False otherwise
    """
    # This is a placeholder - implement actual authentication logic
    return bool(token and len(token) > 0)