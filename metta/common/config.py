"""Base configuration classes using Pydantic."""

from pydantic import BaseModel


class Config(BaseModel):
    """Base configuration class for all Metta configs.

    Provides standard serialization, validation, and configuration
    management using Pydantic.
    """

    class Config:
        """Pydantic configuration."""

        # Allow extra fields for extensibility
        extra = "forbid"

        # Use enum values in serialization
        use_enum_values = True
