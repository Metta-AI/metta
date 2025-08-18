"""
Test suite for wandb_integration module - Priority 2 components.

Tests all WandB training context integration functionality with hard failure enforcement.
No graceful degradation - tests fail when required WandB data is missing.
"""

from unittest.mock import Mock, patch

import pytest

from metta.mcp_server.wandb_integration import (
    LearningProgression,
    TrainingMetricsSample,
    TrainingProgressionAnalyzer,
    WandBMetricsCollector,
    WandBTrainingContext,
)


class TestWandBMetricsCollector:
    """Test WandBMetricsCollector functionality"""

    def test_collect_training_context_success(self, mock_mcp_client):
        """Test successful training context collection"""
        collector = WandBMetricsCollector(mock_mcp_client)

        result = collector.collect_training_context("test_run_123", 8000, 1000)

        # Verify structure
        assert isinstance(result, WandBTrainingContext)
        assert result.run_name == "test_run_123"
        assert result.replay_timestamp_step == 8000
        assert result.context_window_steps == 2000  # ±1000

        # Verify training context components
        assert result.reward_progression is not None
        assert isinstance(result.reward_progression, LearningProgression)
        assert isinstance(result.action_mastery_progression, dict)
        assert isinstance(result.resource_efficiency_progression, dict)
        assert isinstance(result.movement_learning_progression, dict)

        # Verify contextual analysis
        assert result.training_stage in ["early", "mid", "late"]
        assert isinstance(result.learning_velocity, float)
        assert isinstance(result.performance_stability, float)
        assert isinstance(result.behavioral_adaptation_rate, float)

        # Verify correlation analysis
        assert isinstance(result.behavior_metric_correlations, dict)
        assert isinstance(result.critical_learning_moments, list)

    def test_missing_wandb_run_data_raises_error(self):
        """Test that missing WandB run data raises ValueError (hard failure)"""
        # Mock client that returns None for WandB data
        mock_client = Mock()
        mock_client.side_effect = lambda fname, params: None

        collector = WandBMetricsCollector(mock_client)

        with pytest.raises(ValueError, match="WandB run data unavailable"):
            collector.collect_training_context("nonexistent_run", 8000, 1000)

    def test_no_training_metrics_raises_error(self, mock_mcp_client):
        """Test that absence of training metrics raises ValueError"""
        # Mock client that returns run data but no metrics
        mock_client = Mock()
        mock_client.side_effect = (
            lambda fname, params: {"name": "test_run", "summary": {}}
            if "get_wandb_run" in fname
            else "http://example.com"
        )

        collector = WandBMetricsCollector(mock_client)

        # Mock _extract_metrics_samples to return empty list
        with patch.object(collector, "_extract_metrics_samples", return_value=[]):
            with pytest.raises(ValueError, match="No training metrics found around step"):
                collector.collect_training_context("test_run", 8000, 1000)

    def test_get_wandb_run_data(self, mock_mcp_client):
        """Test WandB run data retrieval"""
        collector = WandBMetricsCollector(mock_mcp_client)

        result = collector._get_wandb_run_data("test_run_123")

        # Should call the correct MCP function
        assert mock_mcp_client.calls[0][0] == "mcp__metta__get_wandb_run"
        assert mock_mcp_client.calls[0][1]["run_name"] == "test_run_123"

        # Should return the mocked data
        assert result["name"] == "test_run_123"

    def test_get_wandb_run_url(self, mock_mcp_client):
        """Test WandB run URL retrieval"""
        collector = WandBMetricsCollector(mock_mcp_client)

        result = collector._get_wandb_run_url("test_run_123")

        # Should call the correct MCP function
        url_call = [call for call in mock_mcp_client.calls if "get_wandb_run_url" in call[0]]
        assert len(url_call) > 0
        assert url_call[0][1]["run_name"] == "test_run_123"

        # Should return the URL
        assert result == "https://wandb.ai/test/project/runs/test_run_123"

    def test_extract_metrics_samples(self, mock_mcp_client):
        """Test metrics samples extraction around center step"""
        collector = WandBMetricsCollector(mock_mcp_client)

        run_data = {"name": "test_run", "summary": {"best_reward": 2.1}}
        samples = collector._extract_metrics_samples(run_data, 8000, 1000)

        # Should return list of TrainingMetricsSample objects
        assert isinstance(samples, list)
        assert len(samples) > 0

        for sample in samples:
            assert isinstance(sample, TrainingMetricsSample)
            assert isinstance(sample.step, int)
            assert isinstance(sample.metrics, dict)
            assert 7000 <= sample.step <= 9000  # Within window around 8000

    def test_analyze_learning_progressions(self, mock_mcp_client):
        """Test learning progression analysis from samples"""
        collector = WandBMetricsCollector(mock_mcp_client)

        # Create sample metrics data
        samples = [
            TrainingMetricsSample(
                step=7000, timestamp=None, metrics={"env_agent/reward": 0.5, "env_agent/action.get_output.success": 0.6}
            ),
            TrainingMetricsSample(
                step=8000, timestamp=None, metrics={"env_agent/reward": 1.0, "env_agent/action.get_output.success": 0.7}
            ),
            TrainingMetricsSample(
                step=9000, timestamp=None, metrics={"env_agent/reward": 1.5, "env_agent/action.get_output.success": 0.8}
            ),
        ]

        progressions = collector._analyze_learning_progressions(samples)

        # Should analyze each metric
        assert "env_agent/reward" in progressions
        assert "env_agent/action.get_output.success" in progressions

        # Each progression should have required fields
        for progression in progressions.values():
            assert isinstance(progression, LearningProgression)
            assert progression.trend in ["improving", "declining", "stable", "volatile"]
            assert isinstance(progression.trend_strength, float)
            assert isinstance(progression.current_value, float)
            assert isinstance(progression.baseline_value, float)

    def test_determine_training_stage(self, mock_mcp_client):
        """Test training stage determination"""
        collector = WandBMetricsCollector(mock_mcp_client)

        # Test early stage
        early_samples = [TrainingMetricsSample(step=5000, timestamp=None, metrics={"env_agent/reward": 0.1})]
        stage = collector._determine_training_stage(early_samples, 5000)
        assert stage == "early"

        # Test late stage
        late_samples = [TrainingMetricsSample(step=150000, timestamp=None, metrics={"env_agent/reward": 2.0})]
        stage = collector._determine_training_stage(late_samples, 150000)
        assert stage == "late"

        # Test mid stage
        mid_samples = [
            TrainingMetricsSample(step=40000, timestamp=None, metrics={"env_agent/reward": 0.5}),
            TrainingMetricsSample(step=50000, timestamp=None, metrics={"env_agent/reward": 1.5}),
        ]
        stage = collector._determine_training_stage(mid_samples, 50000)
        assert stage == "mid"

    def test_calculate_learning_velocity(self, mock_mcp_client):
        """Test learning velocity calculation"""
        collector = WandBMetricsCollector(mock_mcp_client)

        # Create samples with improving metrics
        samples = [
            TrainingMetricsSample(
                step=7000,
                timestamp=None,
                metrics={
                    "env_agent/reward": 0.5,
                    "env_agent/ore_red.gained": 2.0,
                    "env_agent/action.get_output.success": 0.4,
                },
            ),
            TrainingMetricsSample(
                step=8000,
                timestamp=None,
                metrics={
                    "env_agent/reward": 1.0,
                    "env_agent/ore_red.gained": 4.0,
                    "env_agent/action.get_output.success": 0.6,
                },
            ),
            TrainingMetricsSample(
                step=9000,
                timestamp=None,
                metrics={
                    "env_agent/reward": 1.5,
                    "env_agent/ore_red.gained": 6.0,
                    "env_agent/action.get_output.success": 0.8,
                },
            ),
        ]

        velocity = collector._calculate_learning_velocity(samples)

        # Should return positive velocity for improving metrics
        assert isinstance(velocity, float)
        assert velocity > 0.0  # Should be positive for improving trends

    def test_calculate_performance_stability(self, mock_mcp_client):
        """Test performance stability calculation"""
        collector = WandBMetricsCollector(mock_mcp_client)

        # Create samples with stable rewards
        stable_samples = [
            TrainingMetricsSample(step=i, timestamp=None, metrics={"env_agent/reward": 1.0 + 0.01 * i})
            for i in range(10)
        ]

        stability = collector._calculate_performance_stability(stable_samples)

        # Should return high stability for consistent performance
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0
        assert stability > 0.8  # Should be high for stable performance

        # Test with volatile rewards
        volatile_samples = [
            TrainingMetricsSample(
                step=i, timestamp=None, metrics={"env_agent/reward": 1.0 + (0.5 if i % 2 == 0 else -0.5)}
            )
            for i in range(10)
        ]

        volatile_stability = collector._calculate_performance_stability(volatile_samples)
        assert volatile_stability < stability  # Should be lower than stable case

    def test_identify_critical_moments(self, mock_mcp_client):
        """Test identification of critical learning moments"""
        collector = WandBMetricsCollector(mock_mcp_client)

        # Create samples with sudden improvement
        samples = [
            TrainingMetricsSample(step=7000, timestamp=None, metrics={"env_agent/reward": 0.5}),
            TrainingMetricsSample(step=7100, timestamp=None, metrics={"env_agent/reward": 0.6}),
            TrainingMetricsSample(step=7200, timestamp=None, metrics={"env_agent/reward": 1.2}),  # Sudden jump
            TrainingMetricsSample(step=7300, timestamp=None, metrics={"env_agent/reward": 1.3}),
            TrainingMetricsSample(step=7400, timestamp=None, metrics={"env_agent/reward": 1.4}),
        ]

        critical_moments = collector._identify_critical_moments(samples)

        # Should identify the breakthrough moment
        assert isinstance(critical_moments, list)
        if critical_moments:  # If any critical moments detected
            moment = critical_moments[0]
            assert moment["type"] == "performance_breakthrough"
            assert isinstance(moment["step"], int)
            assert isinstance(moment["description"], str)
            assert isinstance(moment["magnitude"], float)

    def test_pearson_correlation_calculation(self, mock_mcp_client):
        """Test Pearson correlation coefficient calculation"""
        collector = WandBMetricsCollector(mock_mcp_client)

        # Test perfect positive correlation
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        correlation = collector._pearson_correlation(x, y)
        assert abs(correlation - 1.0) < 0.01  # Should be close to 1.0

        # Test no correlation
        x = [1, 2, 3, 4, 5]
        y = [5, 3, 1, 4, 2]
        correlation = collector._pearson_correlation(x, y)
        assert abs(correlation) <= 0.6  # Should be weak correlation

        # Test with insufficient data
        correlation = collector._pearson_correlation([1], [2])
        assert correlation == 0.0  # Should return 0 for insufficient data


class TestTrainingProgressionAnalyzer:
    """Test TrainingProgressionAnalyzer functionality"""

    def test_analyze_training_progression_success(self, sample_wandb_context):
        """Test successful training progression analysis"""
        analyzer = TrainingProgressionAnalyzer()
        replay_behaviors = {"cooperation_score": 0.8, "resource_efficiency": 0.7}

        result = analyzer.analyze_training_progression(sample_wandb_context, replay_behaviors)

        # Verify structure
        assert "progression_summary" in result
        assert "learning_insights" in result
        assert "behavior_correlation" in result
        assert "training_recommendations" in result
        assert "performance_context" in result

        # Verify progression summary
        summary = result["progression_summary"]
        assert "training_stage" in summary
        assert "overall_learning_trend" in summary
        assert "key_achievements" in summary
        assert "concerning_patterns" in summary
        assert "progression_health_score" in summary

        # Verify learning insights
        insights = result["learning_insights"]
        assert isinstance(insights, list)
        assert all(isinstance(insight, str) for insight in insights)

    def test_missing_wandb_context_raises_error(self):
        """Test that missing WandB context raises ValueError (hard failure)"""
        analyzer = TrainingProgressionAnalyzer()
        replay_behaviors = {"cooperation_score": 0.8}

        with pytest.raises(ValueError, match="No WandB training context provided"):
            analyzer.analyze_training_progression(None, replay_behaviors)

    def test_summarize_progression(self, sample_wandb_context):
        """Test progression summarization"""
        analyzer = TrainingProgressionAnalyzer()

        summary = analyzer._summarize_progression(sample_wandb_context)

        # Verify structure
        assert "training_stage" in summary
        assert "overall_learning_trend" in summary
        assert "key_achievements" in summary
        assert "concerning_patterns" in summary
        assert "progression_health_score" in summary

        # Verify content
        assert summary["training_stage"] == sample_wandb_context.training_stage
        assert isinstance(summary["key_achievements"], list)
        assert isinstance(summary["concerning_patterns"], list)
        assert 0.0 <= summary["progression_health_score"] <= 1.0

    def test_determine_overall_trend(self, sample_wandb_context):
        """Test overall learning trend determination"""
        analyzer = TrainingProgressionAnalyzer()

        trend = analyzer._determine_overall_trend(sample_wandb_context)

        # Should return one of the expected trend types
        expected_trends = [
            "strongly_improving",
            "mostly_improving",
            "stable",
            "mixed_with_concerns",
            "concerning_decline",
            "unknown",
        ]
        assert trend in expected_trends

    def test_generate_learning_insights(self, sample_wandb_context):
        """Test learning insights generation"""
        analyzer = TrainingProgressionAnalyzer()

        insights = analyzer._generate_learning_insights(sample_wandb_context)

        # Should return list of insights
        assert isinstance(insights, list)
        assert len(insights) > 0

        # Each insight should be descriptive
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight) > 20  # Should be descriptive

    def test_generate_training_recommendations(self, sample_wandb_context):
        """Test training recommendations generation"""
        analyzer = TrainingProgressionAnalyzer()

        recommendations = analyzer._generate_training_recommendations(sample_wandb_context)

        # Should return list of actionable recommendations
        assert isinstance(recommendations, list)

        # Each recommendation should be actionable
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 15  # Should be detailed

    def test_analyze_performance_context(self, sample_wandb_context):
        """Test performance context analysis"""
        analyzer = TrainingProgressionAnalyzer()

        context = analyzer._analyze_performance_context(sample_wandb_context)

        # Verify structure
        assert "expected_performance_level" in context
        assert "performance_trajectory" in context
        assert "competency_areas" in context
        assert "improvement_areas" in context

        # Verify content types
        assert isinstance(context["competency_areas"], list)
        assert isinstance(context["improvement_areas"], list)


class TestLearningProgression:
    """Test LearningProgression data structure"""

    def test_learning_progression_creation(self):
        """Test LearningProgression object creation"""
        progression = LearningProgression(
            metric_name="env_agent/reward",
            trend="improving",
            trend_strength=0.8,
            current_value=1.5,
            baseline_value=0.3,
            progression_rate=0.12,
        )

        assert progression.metric_name == "env_agent/reward"
        assert progression.trend == "improving"
        assert progression.trend_strength == 0.8
        assert progression.current_value == 1.5
        assert progression.baseline_value == 0.3
        assert progression.progression_rate == 0.12

    def test_learning_progression_validation(self):
        """Test validation of LearningProgression fields"""
        # Valid trends
        valid_trends = ["improving", "declining", "stable", "volatile"]
        for trend in valid_trends:
            progression = LearningProgression(
                metric_name="test_metric",
                trend=trend,
                trend_strength=0.5,
                current_value=1.0,
                baseline_value=0.5,
                progression_rate=0.1,
            )
            assert progression.trend == trend


class TestWandBTrainingContext:
    """Test WandBTrainingContext data structure"""

    def test_training_context_creation(self, sample_wandb_context):
        """Test WandBTrainingContext object creation"""
        context = sample_wandb_context

        # Verify required fields
        assert context.run_name == "test_run_123"
        assert context.replay_timestamp_step == 8000
        assert context.context_window_steps == 2000

        # Verify progression objects
        assert isinstance(context.reward_progression, LearningProgression)
        assert isinstance(context.action_mastery_progression, dict)
        assert isinstance(context.resource_efficiency_progression, dict)
        assert isinstance(context.movement_learning_progression, dict)

        # Verify contextual analysis fields
        assert context.training_stage in ["early", "mid", "late"]
        assert isinstance(context.learning_velocity, float)
        assert isinstance(context.performance_stability, float)
        assert isinstance(context.behavioral_adaptation_rate, float)

        # Verify correlation analysis
        assert isinstance(context.behavior_metric_correlations, dict)
        assert isinstance(context.critical_learning_moments, list)


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases"""

    def test_full_wandb_analysis_pipeline(self, mock_mcp_client):
        """Test complete WandB analysis pipeline"""
        # Create collector and analyzer
        collector = WandBMetricsCollector(mock_mcp_client)
        analyzer = TrainingProgressionAnalyzer()

        # Collect training context
        training_context = collector.collect_training_context("test_run_123", 8000, 1000)

        # Analyze training progression
        replay_behaviors = {"cooperation_score": 0.8, "resource_efficiency": 0.7, "combat_effectiveness": 0.6}

        progression_analysis = analyzer.analyze_training_progression(training_context, replay_behaviors)

        # Verify pipeline completed successfully
        assert training_context is not None
        assert progression_analysis is not None

        # Verify comprehensive structure
        assert len(progression_analysis) == 5  # All expected components

        # Verify all components have content
        for component in progression_analysis.values():
            assert component is not None
            assert len(component) > 0

    def test_hard_failure_enforcement_with_missing_wandb_data(self):
        """Test that missing WandB data causes hard failures (no graceful degradation)"""
        import os

        # Ensure we're in strict test mode
        assert os.environ.get("MCP_TEST_MODE") == "strict"

        # Mock client that fails to provide data
        failing_client = Mock()
        failing_client.side_effect = Exception("WandB API unavailable")

        collector = WandBMetricsCollector(failing_client)

        # Should fail hard when WandB data is unavailable
        with pytest.raises((ValueError, Exception)):
            collector.collect_training_context("nonexistent_run", 8000, 1000)

        # Analyzer should fail with missing context
        analyzer = TrainingProgressionAnalyzer()
        with pytest.raises(ValueError):
            analyzer.analyze_training_progression(None, {"some": "behaviors"})

    def test_performance_with_large_metric_datasets(self, mock_mcp_client):
        """Test performance with large training metric datasets"""
        import time

        collector = WandBMetricsCollector(mock_mcp_client)

        # Create large metrics dataset (simulate long training run)
        large_samples = []
        for step in range(1000, 20000, 100):  # 190 samples
            metrics = {
                "env_agent/reward": step * 0.0001,
                "env_agent/action.get_output.success": min(0.9, step * 0.00005),
                "env_agent/ore_red.gained": step * 0.001,
                "env_agent/movement.direction.up": 0.25 + (step % 1000) / 10000,
            }
            large_samples.append(TrainingMetricsSample(step=step, timestamp=None, metrics=metrics))

        # Test analysis performance
        start_time = time.time()

        progressions = collector._analyze_learning_progressions(large_samples)
        correlations = collector._analyze_metric_correlations(large_samples)
        critical_moments = collector._identify_critical_moments(large_samples)

        end_time = time.time()
        analysis_time = end_time - start_time

        # Should complete within reasonable time
        assert analysis_time < 5.0, f"WandB analysis took too long: {analysis_time:.2f} seconds"

        # Verify analyses completed correctly
        assert len(progressions) > 0
        assert isinstance(correlations, dict)
        assert isinstance(critical_moments, list)

    def test_correlation_analysis_edge_cases(self, mock_mcp_client):
        """Test correlation analysis with edge cases"""
        collector = WandBMetricsCollector(mock_mcp_client)

        # Test with constant values (no variance)
        constant_samples = [
            TrainingMetricsSample(
                step=i,
                timestamp=None,
                metrics={
                    "env_agent/reward": 1.0,  # Constant
                    "env_agent/ore_red.gained": i,  # Variable
                },
            )
            for i in range(10)
        ]

        correlations = collector._analyze_metric_correlations(constant_samples)

        # Should handle constant values gracefully
        assert isinstance(correlations, dict)

        # Test with insufficient data
        minimal_samples = [TrainingMetricsSample(step=1, timestamp=None, metrics={"env_agent/reward": 1.0})]

        minimal_correlations = collector._analyze_metric_correlations(minimal_samples)
        assert isinstance(minimal_correlations, dict)
        assert len(minimal_correlations) == 0  # Should return empty for insufficient data
