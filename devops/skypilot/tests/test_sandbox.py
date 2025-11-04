#!/usr/bin/env -S uv run pytest
"""Unit tests for sandbox.py functionality."""

# Import the function we want to test
# Note: We need to handle the import carefully since sandbox.py is in recipes/
import sys
from pathlib import Path

import pytest

# Add the parent directory to the path so we can import from recipes
sys.path.insert(0, str(Path(__file__).parent.parent))

from recipes.sandbox import get_cpu_instance_info


class TestCPUInstanceInfo:
    """Tests for get_cpu_instance_info function."""

    def test_exact_core_match(self):
        """Test that exact core counts map to correct instances."""
        test_cases = [
            (2, "c6i.large", 2, 4),
            (8, "c6i.2xlarge", 8, 16),
            (32, "c6i.8xlarge", 32, 64),
            (64, "c6i.16xlarge", 64, 128),
        ]

        for cores, expected_instance, expected_cores, expected_ram in test_cases:
            instance_type, region, _cost, actual_cores, ram_gb = get_cpu_instance_info(cores, cloud="aws")
            assert instance_type == expected_instance, f"Expected {expected_instance} for {cores} cores"
            assert actual_cores == expected_cores, f"Expected {expected_cores} actual cores"
            assert ram_gb == expected_ram, f"Expected {expected_ram}GB RAM"
            assert region == "us-east-1"

    def test_round_up_cores(self):
        """Test that non-standard core counts round up to next available size."""
        test_cases = [
            (3, "c6i.xlarge", 4),  # 3 rounds up to 4
            (10, "c6i.4xlarge", 16),  # 10 rounds up to 16
            (40, "c6i.12xlarge", 48),  # 40 rounds up to 48
            (50, "c6i.16xlarge", 64),  # 50 rounds up to 64
        ]

        for requested_cores, expected_instance, expected_actual_cores in test_cases:
            instance_type, _region, _cost, actual_cores, _ram_gb = get_cpu_instance_info(requested_cores, cloud="aws")
            assert instance_type == expected_instance, f"Expected {expected_instance} for {requested_cores} cores"
            assert actual_cores == expected_actual_cores, f"Expected to round up to {expected_actual_cores} cores"

    def test_maximum_cores(self):
        """Test that requesting more than max cores gives largest instance."""
        instance_type, _region, _cost, actual_cores, ram_gb = get_cpu_instance_info(200, cloud="aws")
        assert instance_type == "c6i.24xlarge"
        assert actual_cores == 96
        assert ram_gb == 192

    def test_minimum_cores(self):
        """Test that requesting 1 core rounds up to 2 (smallest c6i instance)."""
        instance_type, _region, _cost, actual_cores, ram_gb = get_cpu_instance_info(1, cloud="aws")
        assert instance_type == "c6i.large"
        assert actual_cores == 2
        assert ram_gb == 4

    def test_ram_scaling(self):
        """Test that RAM scales correctly with cores (2GB per vCPU for c6i)."""
        test_cases = [
            (2, 4),  # 2 cores = 4GB
            (16, 32),  # 16 cores = 32GB
            (32, 64),  # 32 cores = 64GB
            (64, 128),  # 64 cores = 128GB
        ]

        for cores, expected_ram in test_cases:
            _instance, _region, _cost, actual_cores, ram_gb = get_cpu_instance_info(cores, cloud="aws")
            assert ram_gb == expected_ram, f"Expected {expected_ram}GB RAM for {actual_cores} cores"
            assert ram_gb == actual_cores * 2, "RAM should be 2GB per vCPU for c6i instances"

    def test_non_aws_cloud(self):
        """Test that non-AWS clouds return appropriate warning values."""
        instance_type, region, cost, cores, ram = get_cpu_instance_info(64, cloud="gcp")
        assert instance_type is None
        assert region == "us-east-1"  # Default region
        assert cost is None
        assert cores == 64  # Should preserve requested cores
        assert ram is None

    def test_custom_region(self):
        """Test that custom region is preserved."""
        _instance, region, _cost, _cores, _ram = get_cpu_instance_info(32, region="us-west-2", cloud="aws")
        assert region == "us-west-2"

    def test_all_valid_sizes(self):
        """Test all valid c6i instance sizes."""
        valid_sizes = [2, 4, 8, 16, 32, 48, 64, 96]
        expected_instances = [
            "c6i.large",
            "c6i.xlarge",
            "c6i.2xlarge",
            "c6i.4xlarge",
            "c6i.8xlarge",
            "c6i.12xlarge",
            "c6i.16xlarge",
            "c6i.24xlarge",
        ]

        for cores, expected_instance in zip(valid_sizes, expected_instances, strict=True):
            instance_type, _region, _cost, actual_cores, _ram = get_cpu_instance_info(cores, cloud="aws")
            assert instance_type == expected_instance
            assert actual_cores == cores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
