"""
Tests for the resolver module
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from mettagrid.resolvers import (
    oc_add,
    oc_choose,
    oc_clamp,
    oc_divide,
    oc_equals,
    oc_if,
    oc_make_integer,
    oc_multiply,
    oc_scaled_range,
    oc_subtract,
    oc_to_odd_min3,
    oc_uniform,
    register_resolvers,
)


def test_if_resolver():
    """Test the if resolver functionality"""
    assert oc_if(True, "yes", "no") == "yes"
    assert oc_if(False, "yes", "no") == "no"


def test_uniform_resolver():
    """Test the uniform resolver generates values within range"""
    # Run multiple times to ensure range is respected
    for _ in range(100):
        val = oc_uniform(10, 20)
        assert 10 <= val <= 20


def test_choose_resolver():
    """Test the choose resolver picks from provided options"""
    choices = [1, 2, 3]
    # Run multiple times to ensure it's working
    results = [oc_choose(*choices) for _ in range(100)]
    # Check that we get all possible choices
    assert all(r in choices for r in results)
    # Check that we get some variety (this could theoretically fail but is very unlikely)
    assert len(set(results)) > 1


def test_arithmetic_resolvers():
    """Test basic arithmetic resolvers"""
    assert oc_add(3, 4) == 7
    assert oc_subtract(10, 3) == 7
    assert oc_multiply(3, 4) == 12
    assert oc_divide(12, 4) == 3


def test_equals_resolver():
    """Test equals resolver comparison"""
    assert oc_equals(5, 5) is True
    assert oc_equals(5, "5") is False
    assert oc_equals("abc", "abc") is True


def test_clamp_resolver():
    """Test clamp resolver bounds checking"""
    assert oc_clamp(15, 10, 20) == 15  # Within range
    assert oc_clamp(5, 10, 20) == 10  # Below min
    assert oc_clamp(25, 10, 20) == 20  # Above max


def test_make_integer_resolver():
    """Test integer conversion resolver"""
    assert oc_make_integer(3.2) == 3
    assert oc_make_integer(3.7) == 4
    assert oc_make_integer(3.5) == 4  # Rounds up at .5


def test_to_odd_min3_resolver():
    """Test conversion to odd number with minimum 3"""
    assert oc_to_odd_min3(4) == 5  # Even becomes next odd
    assert oc_to_odd_min3(5) == 5  # Odd stays the same
    assert oc_to_odd_min3(1) == 3  # Below 3 becomes 3
    assert oc_to_odd_min3(2) == 3  # Below 3 becomes 3
    assert oc_to_odd_min3(0) == 3  # Zero becomes 3


class TestScaledRange:
    """Tests for the sampling (scaled range) resolver"""

    def test_deterministic_mode(self):
        """When sampling is 0, should return exactly the center value"""
        root = {"sampling": 0}
        # Integer case
        assert oc_scaled_range(1, 100, 50, root=root) == 50
        # Float case
        assert oc_scaled_range(1.0, 100.0, 50.5, root=root) == 50.5

    def test_full_range(self):
        """When sampling is 1, should use the full range"""
        np.random.seed(42)  # For reproducibility
        root = {"sampling": 1.0}

        # Run multiple times and check that values fall in correct range
        values = [oc_scaled_range(1, 100, 50, root=root) for _ in range(100)]
        # Should be integers due to center being an integer
        assert all(isinstance(v, int) for v in values)
        # Should be within the full range
        assert all(1 <= v <= 100 for v in values)

        # Similar test for float center
        float_values = [oc_scaled_range(1.0, 100.0, 50.5, root=root) for _ in range(100)]
        # Should be floats due to center being a float
        assert all(isinstance(v, float) for v in float_values)
        # Should be within the full range
        assert all(1.0 <= v <= 100.0 for v in float_values)

    def test_partial_range(self):
        """When sampling is between 0 and 1, should use partial range"""
        np.random.seed(42)  # For reproducibility
        root = {"sampling": 0.5}

        # With sampling=0.5, values should be in range [center-0.5*left, center+0.5*right]
        # For range [1, 100] with center 50, this means [25.5, 74.5]
        values = [oc_scaled_range(1, 100, 50, root=root) for _ in range(100)]
        assert all(25 <= v <= 75 for v in values)

    def test_sampling_validation(self):
        """Should raise an assertion error if sampling > 1"""
        root = {"sampling": 1.1}
        with pytest.raises(AssertionError):
            oc_scaled_range(1, 100, 50, root=root)


def test_resolver_registration():
    """Test that resolvers are properly registered with OmegaConf"""
    # Register resolvers directly in this test
    register_resolvers()

    conf = OmegaConf.create(
        {
            "add_result": "${add:2,3}",
            "sub_result": "${sub:10,5}",
            "if_result": "${if:true,'yes','no'}",
            "eq_result": "${eq:'test','test'}",
        }
    )

    # Explicitly resolve all interpolations
    OmegaConf.resolve(conf)

    assert OmegaConf.to_container(conf) == {
        "add_result": 5,
        "sub_result": 5,
        "if_result": "yes",
        "eq_result": True,
    }


def test_sampling_resolver_in_config():
    """Test the sampling resolver in a configuration scenario"""
    # Register resolvers directly in this test
    register_resolvers()

    # Create a config with the sampling parameter
    conf = OmegaConf.create(
        {
            "sampling": 0,  # Deterministic mode
            "param1": "${sampling:1,100,50}",
            "param2": "${sampling:1,100,25}",
        }
    )

    # Explicitly resolve all interpolations
    OmegaConf.resolve(conf)

    # In deterministic mode, should get exact center values
    assert OmegaConf.to_container(conf) == {
        "sampling": 0,
        "param1": 50,
        "param2": 25,
    }

    # Change to random mode
    conf.sampling = 1.0
    resolved = OmegaConf.to_container(conf)

    # Values should now be randomized, but within range
    assert 1 <= resolved["param1"] <= 100
    assert 1 <= resolved["param2"] <= 100
