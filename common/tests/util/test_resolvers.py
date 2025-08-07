"""Tests for metta.common.util.resolvers module."""

import datetime
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from metta.common.util.resolvers import (
    ResolverRegistrar,
    oc_add,
    oc_clamp,
    oc_date_format,
    oc_divide,
    oc_equals,
    oc_if,
    oc_iir,
    oc_make_integer,
    oc_scale,
    oc_to_odd_min3,
    register_resolvers,
)


class TestBasicResolvers:
    """Test basic arithmetic and logical resolvers."""

    def test_oc_if_conditions(self):
        """Test oc_if with various conditions."""
        assert oc_if(True, "yes", "no") == "yes"
        assert oc_if(False, "yes", "no") == "no"
        assert oc_if(5 > 3, 10, 20) == 10

    def test_oc_divide_types(self):
        """Test division with different type handling."""
        assert oc_divide(8, 2) == 4
        assert isinstance(oc_divide(8, 2), int)
        assert oc_divide(7, 2) == 3.5
        assert isinstance(oc_divide(7, 2), float)

    def test_arithmetic_operations(self):
        """Test all arithmetic operations."""
        assert oc_add(5, 3) == 8
        assert oc_clamp(15, 0, 10) == 10
        assert oc_make_integer(5.7) == 6

    def test_comparison_operations(self):
        """Test comparison operations."""
        assert oc_equals(5, 5) is True
        assert oc_equals(5, 6) is False


class TestUtilityResolvers:
    """Test utility resolvers."""

    def test_oc_to_odd_min3(self):
        """Test odd number conversion."""
        assert oc_to_odd_min3(4) == 5
        assert oc_to_odd_min3(0) == 3
        assert oc_to_odd_min3(5) == 5

    def test_oc_scale_basic(self):
        """Test basic scaling."""
        result = oc_scale(5, 0, 10, 0, 100, "linear")
        assert result == 50

    def test_oc_scale_invalid_type(self):
        """Test error handling for invalid scale type."""
        with pytest.raises(ValueError, match="Unknown scale_type: invalid"):
            oc_scale(5, 0, 10, 0, 100, "invalid")

    def test_oc_iir_basic(self):
        """Test IIR filtering."""
        result = oc_iir(0.5, 10, 0)
        assert result == 5.0
        
        # Test integer result preservation
        result = oc_iir(0.5, 10, 6)
        assert isinstance(result, int)
        assert result == 8


class TestDateResolver:
    """Test the oc_date_format resolver."""

    @patch('datetime.datetime')
    def test_oc_date_format_simplified(self, mock_datetime):
        """Test date formatting with simplified codes."""
        mock_now = Mock()
        mock_now.strftime.return_value = "2023-12-25"
        mock_datetime.now.return_value = mock_now

        result = oc_date_format("YYYY-MM-DD")

        mock_datetime.now.assert_called_once()
        mock_now.strftime.assert_called_once_with("%Y-%m-%d")
        assert result == "2023-12-25"

    @patch('datetime.datetime')
    def test_oc_date_format_python_format(self, mock_datetime):
        """Test date formatting with Python format codes."""
        mock_now = Mock()
        mock_now.strftime.return_value = "Monday"
        mock_datetime.now.return_value = mock_now

        result = oc_date_format("%A")

        mock_now.strftime.assert_called_once_with("%A")
        assert result == "Monday"


class TestResolverRegistrar:
    """Test the ResolverRegistrar class."""

    def test_resolver_registrar_init(self):
        """Test ResolverRegistrar initialization."""
        registrar = ResolverRegistrar()
        assert registrar.resolver_count == 0
        assert registrar.logger.name == "ResolverRegistrar"

    @patch('metta.common.util.resolvers.OmegaConf.register_new_resolver')
    def test_register_resolvers(self, mock_register):
        """Test resolver registration."""
        registrar = ResolverRegistrar()
        result = registrar.register_resolvers()

        assert mock_register.call_count > 15
        assert registrar.resolver_count > 15
        assert result is registrar

        mock_register.assert_any_call("if", oc_if, replace=True)
        mock_register.assert_any_call("add", oc_add, replace=True)

    @patch('metta.common.util.resolvers.OmegaConf.register_new_resolver')
    def test_callbacks(self, mock_register):
        """Test callback methods."""
        registrar = ResolverRegistrar()
        config = OmegaConf.create({"test": "value"})

        with patch.object(registrar, 'logger') as mock_logger:
            registrar.on_run_start(config)
            registrar.on_multirun_start(config)

        assert mock_register.call_count > 0
        assert mock_logger.info.call_count == 2

        # on_job_start should do nothing
        registrar.on_job_start(config)

    @patch('metta.common.util.resolvers.ResolverRegistrar')
    def test_legacy_function(self, mock_registrar_class):
        """Test legacy register_resolvers function."""
        mock_instance = Mock()
        mock_registrar_class.return_value = mock_instance

        register_resolvers()

        mock_registrar_class.assert_called_once()
        mock_instance.register_resolvers.assert_called_once()


class TestScaleEdgeCases:
    """Test edge cases for oc_scale."""

    def test_scale_same_output_range(self):
        """Test scaling when output min equals max."""
        result = oc_scale(5, 0, 10, 42, 42, "linear")
        assert result == 42

    def test_scale_assertions(self):
        """Test input validation."""
        with pytest.raises(AssertionError):
            oc_scale(5, 10, 5, 0, 100, "linear")

    def test_scale_zero_handling(self):
        """Test edge case with zero normalization in log scale."""
        result = oc_scale(0, 0, 10, 0, 100, "log")
        assert result == 0

    def test_scale_different_types(self):
        """Test all scale types work."""
        for scale_type in ["linear", "log", "exp", "sigmoid"]:
            result = oc_scale(5, 0, 10, 0, 100, scale_type)
            assert 0 <= result <= 100


class TestIirEdgeCases:
    """Test edge cases for IIR filter."""

    def test_iir_alpha_clamping(self):
        """Test alpha clamping."""
        assert oc_iir(1.5, 10, 0) == 10  # Alpha clamped to 1
        assert oc_iir(-0.5, 10, 5) == 5  # Alpha clamped to 0

    def test_iir_type_preservation(self):
        """Test type preservation in IIR."""
        int_result = oc_iir(0.5, 10, 6)
        assert isinstance(int_result, int)
        
        float_result = oc_iir(0.3, 10.0, 5.0)
        assert isinstance(float_result, float)