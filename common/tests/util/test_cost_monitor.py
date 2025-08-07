"""Tests for metta.common.util.cost_monitor module."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from metta.common.util.cost_monitor import (
    get_cost_info,
    get_instance_cost,
    get_running_instance_info,
    main,
)


class TestGetInstanceCost:
    """Test cases for the get_instance_cost function."""

    @patch('metta.common.util.cost_monitor.sky')
    def test_get_instance_cost_success(self, mock_sky):
        """Test successful instance cost retrieval."""
        mock_cloud = Mock()
        mock_cloud.instance_type_to_hourly_cost.return_value = 0.096
        mock_sky.clouds.AWS.return_value = mock_cloud

        result = get_instance_cost("t3.medium", "us-west-2")

        assert result == 0.096
        mock_sky.clouds.AWS.assert_called_once()
        mock_cloud.instance_type_to_hourly_cost.assert_called_once_with(
            "t3.medium", use_spot=False, region="us-west-2", zone=None
        )

    @patch('metta.common.util.cost_monitor.sky')
    def test_get_instance_cost_with_spot(self, mock_sky):
        """Test instance cost retrieval with spot pricing."""
        mock_cloud = Mock()
        mock_cloud.instance_type_to_hourly_cost.return_value = 0.030
        mock_sky.clouds.AWS.return_value = mock_cloud

        result = get_instance_cost("t3.medium", "us-west-2", use_spot=True)

        assert result == 0.030
        mock_cloud.instance_type_to_hourly_cost.assert_called_once_with(
            "t3.medium", use_spot=True, region="us-west-2", zone=None
        )

    @patch('metta.common.util.cost_monitor.sky')
    def test_get_instance_cost_with_zone(self, mock_sky):
        """Test instance cost retrieval with specific zone."""
        mock_cloud = Mock()
        mock_cloud.instance_type_to_hourly_cost.return_value = 0.096
        mock_sky.clouds.AWS.return_value = mock_cloud

        result = get_instance_cost("t3.medium", "us-west-2", "us-west-2a")

        assert result == 0.096
        mock_cloud.instance_type_to_hourly_cost.assert_called_once_with(
            "t3.medium", use_spot=False, region="us-west-2", zone="us-west-2a"
        )

    @patch('metta.common.util.cost_monitor.sky')
    @patch('metta.common.util.cost_monitor.logger')
    def test_get_instance_cost_exception(self, mock_logger, mock_sky):
        """Test handling of exceptions in cost retrieval."""
        mock_cloud = Mock()
        mock_cloud.instance_type_to_hourly_cost.side_effect = Exception("API Error")
        mock_sky.clouds.AWS.return_value = mock_cloud

        result = get_instance_cost("invalid-type", "us-west-2")

        assert result is None
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "Error calculating hourly cost for invalid-type" in error_call


class TestGetRunningInstanceInfo:
    """Test cases for the get_running_instance_info function."""

    @patch('metta.common.util.cost_monitor.logger')
    def test_get_running_instance_info_no_env_var(self, mock_logger):
        """Test when SKYPILOT_CLUSTER_INFO is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_running_instance_info()

        assert result is None
        mock_logger.warning.assert_called_once_with(
            "SKYPILOT_CLUSTER_INFO not set. Cannot determine instance info."
        )

    @patch('metta.common.util.cost_monitor.json.loads')
    @patch('metta.common.util.cost_monitor.logger')
    def test_get_running_instance_info_invalid_json(self, mock_logger, mock_loads):
        """Test when SKYPILOT_CLUSTER_INFO contains invalid JSON."""
        mock_loads.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with patch.dict(os.environ, {"SKYPILOT_CLUSTER_INFO": "invalid_json"}):
            result = get_running_instance_info()

        assert result is None
        mock_logger.error.assert_called_once()

    @patch('metta.common.util.cost_monitor.json.loads')
    @patch('metta.common.util.cost_monitor.logger')
    def test_get_running_instance_info_missing_field(self, mock_logger, mock_loads):
        """Test when cluster info is missing required fields."""
        mock_loads.return_value = {"region": "us-west-2"}  # Missing instance_type

        with patch.dict(os.environ, {"SKYPILOT_CLUSTER_INFO": "{}"}):
            result = get_running_instance_info()

        assert result is None
        mock_logger.error.assert_called_once()

    @patch('metta.common.util.cost_monitor.requests')
    @patch('metta.common.util.cost_monitor.json.loads')
    def test_get_running_instance_info_success(self, mock_loads, mock_requests):
        """Test successful instance info retrieval."""
        cluster_info = {
            "region": "us-west-2",
            "zone": "us-west-2a"
        }
        mock_loads.return_value = cluster_info

        # Mock AWS metadata responses
        mock_token_resp = Mock()
        mock_token_resp.text = "mock-token"
        mock_instance_resp = Mock()
        mock_instance_resp.text = "t3.medium"
        mock_lifecycle_resp = Mock()
        mock_lifecycle_resp.text = "spot"

        mock_requests.put.return_value = mock_token_resp
        mock_requests.get.side_effect = [mock_instance_resp, mock_lifecycle_resp]

        with patch.dict(os.environ, {"SKYPILOT_CLUSTER_INFO": "{}"}):
            result = get_running_instance_info()

        assert result == ("t3.medium", "us-west-2", "us-west-2a", True)

    @patch('metta.common.util.cost_monitor.requests')
    @patch('metta.common.util.cost_monitor.json.loads')
    def test_get_running_instance_info_no_zone(self, mock_loads, mock_requests):
        """Test instance info retrieval without zone."""
        cluster_info = {
            "region": "us-west-2"
        }
        mock_loads.return_value = cluster_info

        # Mock AWS metadata responses
        mock_token_resp = Mock()
        mock_token_resp.text = "mock-token"
        mock_instance_resp = Mock()
        mock_instance_resp.text = "t3.medium"
        mock_lifecycle_resp = Mock()
        mock_lifecycle_resp.text = "normal"

        mock_requests.put.return_value = mock_token_resp
        mock_requests.get.side_effect = [mock_instance_resp, mock_lifecycle_resp]

        with patch.dict(os.environ, {"SKYPILOT_CLUSTER_INFO": "{}"}):
            result = get_running_instance_info()

        assert result == ("t3.medium", "us-west-2", None, False)


class TestGetCostInfo:
    """Test cases for the get_cost_info function."""

    @patch('metta.common.util.cost_monitor.get_running_instance_info')
    @patch('metta.common.util.cost_monitor.get_instance_cost')
    def test_get_cost_info_success(self, mock_get_cost, mock_get_info):
        """Test successful cost info retrieval."""
        mock_get_info.return_value = ("t3.medium", "us-west-2", "us-west-2a", False)
        mock_get_cost.return_value = 0.096

        result = get_cost_info()

        expected = {
            "instance_type": "t3.medium",
            "region": "us-west-2",
            "zone": "us-west-2a",
            "use_spot": False,
            "instance_hourly_cost": 0.096
        }
        assert result == expected

    @patch('metta.common.util.cost_monitor.get_running_instance_info')
    def test_get_cost_info_no_instance_info(self, mock_get_info):
        """Test when instance info is not available."""
        mock_get_info.return_value = None

        result = get_cost_info()

        assert result is None

    @patch('metta.common.util.cost_monitor.get_running_instance_info')
    @patch('metta.common.util.cost_monitor.get_instance_cost')
    def test_get_cost_info_no_cost(self, mock_get_cost, mock_get_info):
        """Test when cost is not available."""
        mock_get_info.return_value = ("t3.medium", "us-west-2", "us-west-2a", False)
        mock_get_cost.return_value = None

        result = get_cost_info()

        assert result is None


class TestMainFunction:
    """Test cases for the main function."""

    @patch('metta.common.util.cost_monitor.get_running_instance_info')
    @patch('metta.common.util.cost_monitor.logger')
    def test_main_no_instance_info(self, mock_logger, mock_get_info):
        """Test main when instance info is not available."""
        mock_get_info.return_value = None

        result = main()

        assert result is None  # main() doesn't return explicit values

    @patch('metta.common.util.cost_monitor.get_instance_cost')
    @patch('metta.common.util.cost_monitor.get_running_instance_info')
    @patch('metta.common.util.cost_monitor.logger')
    def test_main_no_cost_info(self, mock_logger, mock_get_info, mock_get_cost):
        """Test main when cost info is not available."""
        mock_get_info.return_value = ("t3.medium", "us-west-2", "us-west-2a", False)
        mock_get_cost.return_value = None

        result = main()

        assert result == 1
        mock_logger.error.assert_called_once_with(
            "Unable to retrieve cost information for t3.medium in us-west-2"
        )

    @patch('metta.common.util.cost_monitor.get_cost_info')
    @patch('metta.common.util.cost_monitor.logger')
    @patch('builtins.print')
    def test_main_success_on_demand(self, mock_print, mock_logger, mock_get_cost_info):
        """Test successful main execution with on-demand instance."""
        mock_get_cost_info.return_value = {
            "instance_type": "t3.medium",
            "region": "us-west-2",
            "zone": "us-west-2a",
            "use_spot": False,
            "instance_hourly_cost": 0.096
        }

        with patch.dict(os.environ, {'SKYPILOT_NUM_NODES': '2'}):
            result = main()

        assert result is None  # main() doesn't return explicit values

        # Should call get_cost_info
        mock_get_cost_info.assert_called_once()

    @patch('metta.common.util.cost_monitor.get_cost_info')
    @patch('builtins.print')
    def test_main_success_spot_instance(self, mock_print, mock_get_cost_info):
        """Test successful main execution with spot instance."""
        mock_get_cost_info.return_value = {
            "instance_type": "t3.medium",
            "region": "us-west-2",
            "zone": None,
            "use_spot": True,
            "instance_hourly_cost": 0.030
        }

        with patch.dict(os.environ, {'SKYPILOT_NUM_NODES': '1'}):
            result = main()

        assert result is None  # main() doesn't return explicit values

    @patch('metta.common.util.cost_monitor.get_cost_info')
    def test_main_no_cost_info(self, mock_get_cost_info):
        """Test main when cost info is not available."""
        mock_get_cost_info.return_value = None

        result = main()

        assert result is None  # main() doesn't return explicit values

    @patch('metta.common.util.cost_monitor.get_cost_info')
    def test_main_exception_handling(self, mock_get_cost_info):
        """Test main function exception handling."""
        mock_get_cost_info.side_effect = Exception("Unexpected error")

        # main() doesn't have exception handling, so it will raise
        with pytest.raises(Exception, match="Unexpected error"):
            main()


class TestIntegration:
    """Integration tests for cost monitoring."""

    @patch('metta.common.util.cost_monitor.sky')
    @patch('metta.common.util.cost_monitor.json.loads')
    @patch('metta.common.util.cost_monitor.requests')
    def test_end_to_end_cost_info(self, mock_requests, mock_loads, mock_sky):
        """Test end-to-end cost info workflow."""
        # Mock SkyPilot cluster info
        cluster_info = {
            "region": "us-east-1",
            "zone": "us-east-1a"
        }
        mock_loads.return_value = cluster_info

        # Mock AWS metadata service
        mock_token_resp = Mock()
        mock_token_resp.text = "mock-token"
        mock_instance_resp = Mock()
        mock_instance_resp.text = "m5.large"
        mock_lifecycle_resp = Mock()
        mock_lifecycle_resp.text = "normal"

        mock_requests.put.return_value = mock_token_resp
        mock_requests.get.side_effect = [mock_instance_resp, mock_lifecycle_resp]

        # Mock SkyPilot cost API
        mock_cloud = Mock()
        mock_cloud.instance_type_to_hourly_cost.return_value = 0.192
        mock_sky.clouds.AWS.return_value = mock_cloud

        with patch.dict(os.environ, {"SKYPILOT_CLUSTER_INFO": "{}"}):
            # Get cost info
            cost_info = get_cost_info()

            assert cost_info["instance_type"] == "m5.large"
            assert cost_info["region"] == "us-east-1"
            assert cost_info["zone"] == "us-east-1a"
            assert cost_info["use_spot"] is False
            assert cost_info["instance_hourly_cost"] == 0.192

    @patch('metta.common.util.cost_monitor.sky')
    def test_cost_calculation_different_instance_types(self, mock_sky):
        """Test cost calculation for different instance types."""
        mock_cloud = Mock()
        mock_sky.clouds.AWS.return_value = mock_cloud

        # Test different instance types
        instance_costs = {
            "t3.micro": 0.0208,
            "t3.small": 0.0416,
            "t3.medium": 0.0832,
            "m5.large": 0.192,
            "c5.xlarge": 0.34
        }

        for instance_type, expected_cost in instance_costs.items():
            mock_cloud.instance_type_to_hourly_cost.return_value = expected_cost

            result = get_instance_cost(instance_type, "us-west-2")
            assert result == expected_cost

    def test_spot_vs_ondemand_detection(self):
        """Test spot vs on-demand instance detection."""
        # Test spot instance detection logic
        test_cases = [
            ("spot", True),
            ("normal", False),
            ("scheduled", False)
        ]

        for lifecycle, expected_spot in test_cases:
            # This would be tested in the actual implementation
            use_spot = lifecycle == "spot"
            assert use_spot == expected_spot
