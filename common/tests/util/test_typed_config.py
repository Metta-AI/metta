"""Tests for metta.common.util.typed_config module."""

import pytest
from pydantic import Field, ValidationError

from metta.common.util.typed_config import BaseModelWithForbidExtra


class TestBaseModelWithForbidExtra:
    """Test cases for BaseModelWithForbidExtra class."""

    def test_inheritance_from_base_model(self):
        """Test that BaseModelWithForbidExtra properly inherits from Pydantic BaseModel."""
        from pydantic import BaseModel

        assert issubclass(BaseModelWithForbidExtra, BaseModel)

        # Test basic instantiation works
        instance = BaseModelWithForbidExtra()
        assert isinstance(instance, BaseModel)
        assert isinstance(instance, BaseModelWithForbidExtra)

    def test_forbids_extra_fields_with_simple_subclass(self):
        """Test that extra fields are forbidden when creating a simple subclass."""

        class SimpleConfig(BaseModelWithForbidExtra):
            name: str
            value: int = Field(default=42)

        # Valid instantiation should work
        config = SimpleConfig(name="test")
        assert config.name == "test"
        assert config.value == 42

        # Valid instantiation with all fields should work
        config2 = SimpleConfig(name="test2", value=100)
        assert config2.name == "test2"
        assert config2.value == 100

        # Extra fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            SimpleConfig(name="test", value=42, extra_field="not_allowed")

        error = exc_info.value
        assert "extra_field" in str(error)
        assert "Extra inputs are not permitted" in str(error)

    def test_forbids_extra_fields_with_complex_subclass(self):
        """Test extra field forbidding with a more complex configuration class."""

        class ComplexConfig(BaseModelWithForbidExtra):
            database_url: str
            max_connections: int = Field(default=10, ge=1, le=100)
            enable_ssl: bool = Field(default=True)
            tags: list[str] = Field(default_factory=list)

        # Valid configuration
        config = ComplexConfig(
            database_url="postgresql://localhost/db",
            max_connections=20,
            enable_ssl=False,
            tags=["prod", "primary"]
        )
        assert config.database_url == "postgresql://localhost/db"
        assert config.max_connections == 20
        assert config.enable_ssl is False
        assert config.tags == ["prod", "primary"]

        # Extra field should be rejected
        with pytest.raises(ValidationError) as exc_info:
            ComplexConfig(
                database_url="postgresql://localhost/db",
                timeout=30,  # This extra field should cause error
            )

        error = exc_info.value
        assert "timeout" in str(error)
        assert "Extra inputs are not permitted" in str(error)

    def test_allows_normal_pydantic_validation(self):
        """Test that normal Pydantic validation still works properly."""

        class ValidatedConfig(BaseModelWithForbidExtra):
            positive_number: int = Field(gt=0)
            email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')

        # Valid data should pass
        config = ValidatedConfig(
            positive_number=5,
            email="test@example.com"
        )
        assert config.positive_number == 5
        assert config.email == "test@example.com"

        # Invalid positive_number should fail validation
        with pytest.raises(ValidationError) as exc_info:
            ValidatedConfig(positive_number=-1, email="test@example.com")
        assert "greater than 0" in str(exc_info.value)

        # Invalid email should fail validation
        with pytest.raises(ValidationError) as exc_info:
            ValidatedConfig(positive_number=5, email="invalid-email")
        assert "String should match pattern" in str(exc_info.value)

    def test_config_dict_is_properly_set(self):
        """Test that the model_config is correctly configured."""
        # The config should be set as a class variable
        assert hasattr(BaseModelWithForbidExtra, 'model_config')

        # It should be a ConfigDict with extra="forbid"
        config = BaseModelWithForbidExtra.model_config
        assert config.get('extra') == 'forbid'

    def test_multiple_inheritance_preserves_forbid_extra(self):
        """Test that the forbid extra behavior is preserved in multiple inheritance scenarios."""

        class MixinClass:
            def custom_method(self):
                return "mixin_result"

        class MultiInheritConfig(BaseModelWithForbidExtra, MixinClass):
            data: str

        # Should still forbid extra fields
        with pytest.raises(ValidationError) as exc_info:
            MultiInheritConfig(data="test", extra="not_allowed")
        assert "Extra inputs are not permitted" in str(exc_info.value)

        # Should still allow valid instantiation and mixin methods
        config = MultiInheritConfig(data="test")
        assert config.data == "test"
        assert config.custom_method() == "mixin_result"

    def test_serialization_and_deserialization(self):
        """Test that serialization works correctly with the forbid extra configuration."""

        class SerializableConfig(BaseModelWithForbidExtra):
            name: str
            count: int = 0

        # Create instance
        original = SerializableConfig(name="test", count=5)

        # Serialize to dict
        data = original.model_dump()
        assert data == {"name": "test", "count": 5}

        # Deserialize from dict (should work)
        restored = SerializableConfig(**data)
        assert restored.name == "test"
        assert restored.count == 5

        # Adding extra field to dict should fail during deserialization
        data_with_extra = {"name": "test", "count": 5, "extra": "bad"}
        with pytest.raises(ValidationError):
            SerializableConfig(**data_with_extra)
