"""Unit tests for BaseCollector."""

from unittest.mock import patch

import pytest

from devops.datadog.utils.base import BaseCollector


class MockCollector(BaseCollector):
    """Mock collector for testing BaseCollector functionality."""

    def __init__(self, name="test_collector"):
        super().__init__(name=name)
        self.collect_called = False

    def collect_metrics(self) -> dict[str, float]:
        """Mock metric collection."""
        self.collect_called = True
        return {
            f"{self.name}.test_metric_1": 42.0,
            f"{self.name}.test_metric_2": 100.5,
        }


class TestBaseCollector:
    """Test BaseCollector base class."""

    def test_initialization(self):
        """Test that collector initializes with correct name."""
        collector = MockCollector(name="my_collector")

        assert collector.name == "my_collector"
        assert collector.collect_called is False

    def test_collect_safe_success(self):
        """Test that collect_safe returns metrics on success."""
        collector = MockCollector()

        metrics = collector.collect_safe()

        assert collector.collect_called is True
        assert "test_collector.test_metric_1" in metrics
        assert metrics["test_collector.test_metric_1"] == 42.0
        assert "test_collector.test_metric_2" in metrics
        assert metrics["test_collector.test_metric_2"] == 100.5

    def test_collect_safe_with_exception(self):
        """Test that collect_safe handles exceptions gracefully."""

        class FailingCollector(BaseCollector):
            def collect_metrics(self) -> dict[str, float]:
                raise ValueError("Test error")

        collector = FailingCollector(name="failing")

        # Should not raise, should return empty dict
        metrics = collector.collect_safe()

        assert metrics == {}

    def test_collect_safe_returns_collected_metrics_on_success(self):
        """Test that collect_safe returns collected metrics on successful collection."""
        collector = MockCollector()

        metrics = collector.collect_safe()

        # Should have the mock metrics
        assert "test_collector.test_metric_1" in metrics
        assert metrics["test_collector.test_metric_1"] == 42.0
        assert "test_collector.test_metric_2" in metrics
        assert metrics["test_collector.test_metric_2"] == 100.5

    def test_collect_safe_with_logger_error(self):
        """Test that errors are logged."""

        class FailingCollector(BaseCollector):
            def collect_metrics(self) -> dict[str, float]:
                raise RuntimeError("Critical error")

        collector = FailingCollector(name="test")

        with patch.object(collector.logger, "error") as mock_logger:
            metrics = collector.collect_safe()

            # Verify error was logged
            mock_logger.assert_called_once()
            call_args = mock_logger.call_args[0][0]
            assert "Critical error" in call_args

            # Verify empty dict returned on error
            assert metrics == {}

    def test_empty_metrics_are_valid(self):
        """Test that returning empty dict is valid."""

        class EmptyCollector(BaseCollector):
            def collect_metrics(self) -> dict[str, float]:
                return {}

        collector = EmptyCollector(name="empty")

        metrics = collector.collect_safe()

        # Empty dict is valid
        assert metrics == {}

    def test_logger_has_correct_name(self):
        """Test that logger is configured with collector name."""
        collector = MockCollector(name="my_test_collector")

        assert "my_test_collector" in collector.logger.name

    def test_abstract_collect_metrics_must_be_implemented(self):
        """Test that BaseCollector cannot be instantiated without collect_metrics."""

        # BaseCollector is abstract, but we can test the pattern
        class IncompleteCollector(BaseCollector):
            pass  # Missing collect_metrics implementation

        # This should fail because collect_metrics is abstract
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteCollector(name="incomplete")
