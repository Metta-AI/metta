"""Unit tests for metric decorator and registry."""

from devops.datadog.utils.decorators import clear_registry, collect_all_metrics, get_registered_metrics, metric


class TestMetricDecorator:
    """Test @metric decorator functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()

    def test_metric_decorator_registers_function(self):
        """Test that @metric decorator registers a function."""

        @metric(name="test.metric")
        def my_metric():
            return 42.0

        registered = get_registered_metrics()

        assert "test.metric" in registered
        assert callable(registered["test.metric"])

    def test_metric_decorator_preserves_function(self):
        """Test that decorator doesn't change function behavior."""

        @metric(name="test.value")
        def calculate_value():
            return 123.45

        result = calculate_value()

        assert result == 123.45

    def test_metric_decorator_with_parameters(self):
        """Test that decorated functions can take parameters."""

        @metric(name="test.add")
        def add_numbers(a, b):
            return a + b

        result = add_numbers(5, 3)

        assert result == 8

    def test_multiple_metrics_registration(self):
        """Test that multiple metrics can be registered."""

        @metric(name="metric.one")
        def metric_one():
            return 1.0

        @metric(name="metric.two")
        def metric_two():
            return 2.0

        @metric(name="metric.three")
        def metric_three():
            return 3.0

        registered = get_registered_metrics()

        assert len(registered) == 3
        assert "metric.one" in registered
        assert "metric.two" in registered
        assert "metric.three" in registered

    def test_collect_all_metrics_calls_all_functions(self):
        """Test that collect_all_metrics invokes all registered functions."""

        @metric(name="test.a")
        def metric_a():
            return 10.0

        @metric(name="test.b")
        def metric_b():
            return 20.0

        metrics = collect_all_metrics()

        assert metrics == {"test.a": 10.0, "test.b": 20.0}

    def test_collect_all_metrics_with_object_methods(self):
        """Test that @metric works with class methods."""

        class MyCollector:
            def __init__(self, value):
                self.value = value

            @metric(name="collector.method_metric")
            def get_metric(self):
                return self.value * 2

        obj = MyCollector(value=5)
        obj.get_metric()  # Call to register

        metrics = collect_all_metrics()

        assert "collector.method_metric" in metrics
        assert metrics["collector.method_metric"] == 10.0

    def test_metric_function_raises_exception(self):
        """Test that collect_all_metrics handles exceptions gracefully."""

        @metric(name="failing.metric")
        def failing_metric():
            raise ValueError("Metric calculation failed")

        @metric(name="working.metric")
        def working_metric():
            return 42.0

        # Should not raise, should skip failing metric
        metrics = collect_all_metrics()

        # Should have working metric but not failing one
        assert "working.metric" in metrics
        assert metrics["working.metric"] == 42.0
        # Failing metric should not be in results
        assert "failing.metric" not in metrics

    def test_metric_returns_none(self):
        """Test that None values are excluded from collected metrics."""

        @metric(name="none.metric")
        def none_metric():
            return None

        @metric(name="valid.metric")
        def valid_metric():
            return 100.0

        metrics = collect_all_metrics()

        assert "valid.metric" in metrics
        assert "none.metric" not in metrics

    def test_clear_registry_empties_registered_metrics(self):
        """Test that clear_registry removes all registered metrics."""

        @metric(name="temp.metric")
        def temp_metric():
            return 1.0

        assert len(get_registered_metrics()) == 1

        clear_registry()

        assert len(get_registered_metrics()) == 0

    def test_duplicate_metric_name_overwrites(self):
        """Test that registering same name twice overwrites."""

        @metric(name="dup.metric")
        def first_version():
            return 1.0

        @metric(name="dup.metric")
        def second_version():
            return 2.0

        registered = get_registered_metrics()

        # Should only have one entry
        assert len([k for k in registered if k == "dup.metric"]) == 1

        # Should use the second version
        metrics = collect_all_metrics()
        assert metrics["dup.metric"] == 2.0

    def test_metric_with_complex_return_type(self):
        """Test that non-numeric returns are handled."""

        @metric(name="string.metric")
        def string_metric():
            return "not a number"

        # collect_all_metrics should handle gracefully
        metrics = collect_all_metrics()

        # String values should be excluded
        assert "string.metric" not in metrics

    def test_get_registered_metrics_returns_copy(self):
        """Test that get_registered_metrics returns a copy, not the original."""

        @metric(name="test.metric")
        def test_metric():
            return 1.0

        registered1 = get_registered_metrics()
        registered2 = get_registered_metrics()

        # Should be equal but not same object
        assert registered1 == registered2
        assert registered1 is not registered2
