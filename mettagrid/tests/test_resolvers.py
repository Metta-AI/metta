"""
Tests for the resolver module
"""

from typing import Any, Dict, Union, cast

import numpy as np
import pytest
from omegaconf import OmegaConf

from mettagrid.resolvers import (
    oc_add,
    oc_choose,
    oc_clamp,
    oc_divide,
    oc_equals,
    oc_greater_than,
    oc_greater_than_or_equal,
    oc_if,
    oc_iir,
    oc_less_than,
    oc_less_than_or_equal,
    oc_make_integer,
    oc_multiply,
    oc_scale,
    oc_scaled_range,
    oc_subtract,
    oc_to_odd_min3,
    oc_uniform,
    register_resolvers,
)


# Fixtures
@pytest.fixture
def omega_conf_with_resolvers():
    """Fixture providing an OmegaConf with resolvers registered"""
    register_resolvers()
    return OmegaConf.create(
        {
            "add_result": "${add:2,3}",
            "sub_result": "${sub:10,5}",
            "if_result": "${if:true,'yes','no'}",
            "eq_result": "${eq:'test','test'}",
        }
    )


@pytest.fixture
def omega_conf_with_sampling():
    """Fixture providing an OmegaConf with sampling parameter"""
    register_resolvers()
    return OmegaConf.create(
        {
            "sampling": 0,  # Deterministic mode
            "param1": "${sampling:1,100,50}",
            "param2": "${sampling:1,100,25}",
        }
    )


class TestBasicResolvers:
    """Tests for basic resolver functionality"""

    @pytest.mark.parametrize(
        "condition,true_value,false_value,expected",
        [
            (True, "yes", "no", "yes"),
            (False, "yes", "no", "no"),
            (True, 10, 20, 10),
            (False, 10, 20, 20),
        ],
    )
    def test_if_resolver(self, condition, true_value, false_value, expected):
        """Test the if resolver with various inputs"""
        assert oc_if(condition, true_value, false_value) == expected

    def test_uniform_resolver(self):
        """Test the uniform resolver generates values within range"""
        # Set seed for reproducibility
        np.random.seed(42)

        # Test multiple ranges
        ranges = [(10, 20), (0, 1), (-10, 10)]
        for min_val, max_val in ranges:
            for _ in range(20):  # Run multiple times to ensure range is respected
                val = oc_uniform(min_val, max_val)
                assert min_val <= val <= max_val

    def test_choose_resolver(self):
        """Test the choose resolver picks from provided options"""
        # Set seed for reproducibility
        np.random.seed(42)

        test_cases = [
            [1, 2, 3],
            ["apple", "banana", "cherry"],
            [True, False],
        ]

        for choices in test_cases:
            # Run multiple times to ensure it's working
            results = [oc_choose(*choices) for _ in range(30)]
            # Check that we get all possible choices
            assert all(r in choices for r in results)
            # Check that we get some variety (this could theoretically fail but is very unlikely)
            assert len(set(results)) > 1

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (3, 4, 7),  # Positive integers
            (-3, 4, 1),  # Mixed signs
            (3.5, 4.5, 8.0),  # Floating point
        ],
    )
    def test_add_resolver(self, a, b, expected):
        """Test addition resolver with various inputs"""
        assert oc_add(a, b) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (10, 3, 7),  # Positive integers
            (3, 10, -7),  # Result negative
            (10.5, 3.5, 7.0),  # Floating point
        ],
    )
    def test_subtract_resolver(self, a, b, expected):
        """Test subtraction resolver with various inputs"""
        assert oc_subtract(a, b) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (3, 4, 12),  # Positive integers
            (-3, 4, -12),  # Mixed signs
            (3.5, 2, 7.0),  # Floating point
        ],
    )
    def test_multiply_resolver(self, a, b, expected):
        """Test multiplication resolver with various inputs"""
        assert oc_multiply(a, b) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (12, 4, 3),  # Positive integers, exact division
            (12, 5, 2.4),  # Positive integers, floating point result
            (-12, 4, -3),  # Mixed signs
        ],
    )
    def test_divide_resolver(self, a, b, expected):
        """Test division resolver with various inputs"""
        assert oc_divide(a, b) == expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (5, 5, True),  # Same integers
            (5, "5", False),  # Different types
            ("abc", "abc", True),  # Same strings
            ("abc", "ABC", False),  # Case-sensitive comparison
            (True, True, True),  # Booleans
            ([], [], True),  # Empty collections
        ],
    )
    def test_equals_resolver(self, a, b, expected):
        """Test equals resolver with various inputs"""
        assert oc_equals(a, b) is expected

    @pytest.mark.parametrize(
        "value,min_val,max_val,expected",
        [
            (15, 10, 20, 15),  # Within range
            (5, 10, 20, 10),  # Below min
            (25, 10, 20, 20),  # Above max
            (10, 10, 20, 10),  # At min boundary
            (20, 10, 20, 20),  # At max boundary
        ],
    )
    def test_clamp_resolver(self, value, min_val, max_val, expected):
        """Test clamp resolver with various inputs"""
        assert oc_clamp(value, min_val, max_val) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (3.2, 3),  # Round down
            (3.5, 4),  # Round up at .5
            (3.7, 4),  # Round up
            (0.1, 0),  # Near zero, round down
            (-1.7, -2),  # Negative, round down
            (-1.2, -1),  # Negative, round up
        ],
    )
    def test_make_integer_resolver(self, value, expected):
        """Test integer conversion resolver with various inputs"""
        assert oc_make_integer(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (4, 5),  # Even becomes next odd
            (5, 5),  # Odd stays the same
            (1, 3),  # Below 3 becomes 3
            (2, 3),  # Below 3 becomes 3
            (0, 3),  # Zero becomes 3
            (-1, 3),  # Negative becomes 3
        ],
    )
    def test_to_odd_min3_resolver(self, value, expected):
        """Test conversion to odd number with minimum 3 with various inputs"""
        assert oc_to_odd_min3(value) == expected


class TestComparisonResolvers:
    """Tests for comparison resolver functionality"""

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (5, 3, True),
            (3, 5, False),
            (4, 4, False),
        ],
    )
    def test_greater_than(self, a, b, expected):
        """Test greater than resolver with various inputs"""
        assert oc_greater_than(a, b) is expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (3, 5, True),
            (5, 3, False),
            (4, 4, False),
        ],
    )
    def test_less_than(self, a, b, expected):
        """Test less than resolver with various inputs"""
        assert oc_less_than(a, b) is expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (5, 5, True),
            (6, 5, True),
            (4, 5, False),
        ],
    )
    def test_greater_than_or_equal(self, a, b, expected):
        """Test greater than or equal resolver with various inputs"""
        assert oc_greater_than_or_equal(a, b) is expected

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (5, 5, True),
            (4, 5, True),
            (6, 5, False),
        ],
    )
    def test_less_than_or_equal(self, a, b, expected):
        """Test less than or equal resolver with various inputs"""
        assert oc_less_than_or_equal(a, b) is expected


class TestAdvancedResolvers:
    """Tests for advanced resolver functionality"""

    @pytest.mark.parametrize(
        "alpha,new_value,last_value,expected",
        [
            (0.9, 10.0, 5.0, pytest.approx(9.5)),  # 0.9*10 + 0.1*5
            (0.5, 8.0, 4.0, pytest.approx(6.0)),  # 0.5*8 + 0.5*4
            (1.0, 7.0, 3.0, pytest.approx(7.0)),  # 1.0*7 + 0.0*3
            (0.0, 7.0, 3.0, pytest.approx(3.0)),  # 0.0*7 + 1.0*3
        ],
    )
    def test_iir(self, alpha, new_value, last_value, expected):
        """Test Infinite Impulse Response filter with various inputs"""
        assert oc_iir(alpha, new_value, last_value) == expected

    @pytest.mark.parametrize(
        "value,in_min,in_max,out_min,out_max,scale_type,expected",
        [
            (0.0, 0.0, 1.0, 0.0, 10.0, "linear", 0.0),
            (0.5, 0.0, 1.0, 0.0, 10.0, "linear", 5.0),
            (1.0, 0.0, 1.0, 0.0, 10.0, "linear", 10.0),
            (0.5, 0.0, 1.0, 0.0, 1.0, "sigmoid", pytest.approx(0.5, abs=0.1)),
            (0.5, 0.0, 1.0, 1.0, 100.0, "exp", pytest.approx(24.76, abs=0.5)),
            (0.5, 0.0, 1.0, 1.0, 100.0, "log", pytest.approx(74.26, abs=0.5)),
        ],
    )
    def test_scale(self, value, in_min, in_max, out_min, out_max, scale_type, expected):
        """Test value scaling with different types and ranges"""
        result = oc_scale(value, in_min, in_max, out_min, out_max, scale_type)
        assert result == expected


class TestScaledRange:
    """Tests for the sampling (scaled range) resolver"""

    @pytest.mark.parametrize(
        "min_val,max_val,center,expected",
        [
            (1, 100, 50, 50),  # Integer center
            (1.0, 100.0, 50.5, 50.5),  # Float center
            (-10, 10, 0, 0),  # Zero center
        ],
    )
    def test_deterministic_mode(self, min_val, max_val, center, expected):
        """When sampling is 0, should return exactly the center value"""
        root = cast(Dict[str, Union[int, float]], {"sampling": 0.0})
        assert oc_scaled_range(min_val, max_val, center, _root_=root) == expected

    def test_full_range(self):
        """When sampling is 1, should use the full range"""
        np.random.seed(42)  # For reproducibility
        root = cast(Dict[str, Union[int, float]], {"sampling": 1.0})

        # Test integer ranges
        int_ranges = [(1, 100, 50), (-50, 50, 0), (1, 10, 5)]
        for min_val, max_val, center in int_ranges:
            values = [oc_scaled_range(min_val, max_val, center, _root_=root) for _ in range(30)]
            # Should be integers due to center being an integer
            assert all(isinstance(v, int) for v in values)
            # Should be within the full range
            assert all(min_val <= v <= max_val for v in values)
            # Should have some variety
            assert len(set(values)) > 1

        # Test float ranges
        float_ranges = [(1.0, 100.0, 50.5), (-10.0, 10.0, 0.5)]
        for min_val, max_val, center in float_ranges:
            values = [oc_scaled_range(min_val, max_val, center, _root_=root) for _ in range(30)]
            # Should be floats due to center being a float
            assert all(isinstance(v, float) for v in values)
            # Should be within the full range
            assert all(min_val <= v <= max_val for v in values)
            # Should have some variety
            assert len(set(values)) > 1

    @pytest.mark.parametrize(
        "sampling,min_val,max_val,center,expected_min,expected_max",
        [
            (0.5, 1, 100, 50, 25, 75),  # 50% range around center 50
            (0.25, 0, 100, 50, 37, 63),  # 25% range around center 50
            (0.75, -100, 100, 0, -75, 75),  # 75% range around center 0
        ],
    )
    def test_partial_range(self, sampling, min_val, max_val, center, expected_min, expected_max):
        """When sampling is between 0 and 1, should use partial range"""
        np.random.seed(42)  # For reproducibility
        root = {"sampling": sampling}

        values = [oc_scaled_range(min_val, max_val, center, _root_=root) for _ in range(100)]
        assert all(expected_min <= v <= expected_max for v in values)

    def test_sampling_validation(self):
        """Should raise an assertion error if sampling > 1"""
        invalid_values = [1.1, 2.0, 100]
        for invalid_value in invalid_values:
            root = {"sampling": invalid_value}
            with pytest.raises(AssertionError):
                oc_scaled_range(1, 100, 50, _root_=root)


class TestConfigIntegration:
    """Tests for resolvers integrated with OmegaConf"""

    def test_resolver_registration(self, omega_conf_with_resolvers):
        """Test that resolvers are properly registered with OmegaConf"""
        # Explicitly resolve all interpolations
        OmegaConf.resolve(omega_conf_with_resolvers)

        assert OmegaConf.to_container(omega_conf_with_resolvers) == {
            "add_result": 5,
            "sub_result": 5,
            "if_result": "yes",
            "eq_result": True,
        }

    def test_sampling_resolver_deterministic(self, omega_conf_with_sampling):
        """Test the sampling resolver in deterministic mode"""
        # Explicitly resolve all interpolations
        OmegaConf.resolve(omega_conf_with_sampling)

        # In deterministic mode, should get exact center values
        assert OmegaConf.to_container(omega_conf_with_sampling) == {
            "sampling": 0,
            "param1": 50,
            "param2": 25,
        }

    def test_sampling_resolver_random(self, omega_conf_with_sampling):
        """Test the sampling resolver in random mode"""
        # Change to random mode and set seed
        np.random.seed(42)

        # Update sampling and re-create the interpolated fields
        omega_conf_with_sampling.sampling = 1.0
        omega_conf_with_sampling.param1 = "${sampling:1,100,50}"
        omega_conf_with_sampling.param2 = "${sampling:1,100,25}"

        # Debug prints
        print(f"On entry - {omega_conf_with_sampling}")
        print(f"after overwrite - {omega_conf_with_sampling}")

        # Resolve and check
        OmegaConf.resolve(omega_conf_with_sampling)
        print(f"after resolve - {omega_conf_with_sampling}")

        resolved = OmegaConf.to_container(omega_conf_with_sampling)

        # Make sure resolved is not None
        assert resolved is not None, "OmegaConf.to_container returned None"

        # Type hinting to help the type checker
        from typing import Dict, cast

        resolved_dict = cast(Dict[str, Any], resolved)
        print(f"resolved_dict - {resolved_dict}")

        # Values should now be randomized, but within range
        param1 = resolved_dict.get("param1")
        param2 = resolved_dict.get("param2")

        # Check that the parameters exist
        assert param1 is not None, "param1 not found in resolved config"
        assert param2 is not None, "param2 not found in resolved config"

        # Now perform the range checks with explicit type conversion
        assert 1 <= float(param1) <= 100
        assert 1 <= float(param2) <= 100

        # Different from center values
        assert float(param1) != 50.0 or float(param2) != 25.0
