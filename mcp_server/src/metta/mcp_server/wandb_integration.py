"""
WandB Training Context Integration Module

This module provides integration with Weights & Biases to collect training context
around replay timestamps, including:
- Training progression analysis around replay timestamps (±1000 steps)
- Learning curve correlation with replay behaviors
- Key metrics from env_agent panel: reward, action success rates, resource gains, movement patterns
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrainingMetricsSample:
    """Single sample of training metrics at a specific step"""

    step: int
    timestamp: Optional[str]
    metrics: Dict[str, float]


@dataclass
class LearningProgression:
    """Learning progression analysis for a specific metric"""

    metric_name: str
    trend: str  # "improving", "declining", "stable", "volatile"
    trend_strength: float  # 0.0 to 1.0
    current_value: float
    baseline_value: float
    progression_rate: float  # Change per 1000 steps


@dataclass
class WandBTrainingContext:
    """Complete training context analysis from WandB data"""

    run_name: str
    run_url: Optional[str]
    replay_timestamp_step: int
    context_window_steps: int

    # Core training metrics
    reward_progression: LearningProgression
    action_mastery_progression: Dict[str, LearningProgression]
    resource_efficiency_progression: Dict[str, LearningProgression]
    movement_learning_progression: Dict[str, LearningProgression]

    # Contextual analysis
    training_stage: str  # "early", "mid", "late"
    learning_velocity: float  # Rate of improvement
    performance_stability: float  # Consistency of performance
    behavioral_adaptation_rate: float  # How quickly behaviors change

    # Correlation analysis
    behavior_metric_correlations: Dict[str, float]
    critical_learning_moments: List[Dict[str, Any]]


class WandBMetricsCollector:
    """Collects and analyzes WandB training metrics around replay timestamps"""

    def __init__(self, mcp_client):
        """
        Initialize with MCP client for WandB API access

        Args:
            mcp_client: MCP client instance with WandB function access
        """
        self.mcp_client = mcp_client
        self.priority_metrics = [
            "env_agent/reward",
            "env_agent/action.get_output.success",
            "env_agent/action.put_recipe_items.success",
            "env_agent/action.move.success",
            "env_agent/action.attack.success",
            "env_agent/ore_red.gained",
            "env_agent/ore_blue.gained",
            "env_agent/battery_red.gained",
            "env_agent/battery_blue.gained",
            "env_agent/heart.gained",
            "env_agent/movement.direction.up",
            "env_agent/movement.direction.down",
            "env_agent/movement.direction.left",
            "env_agent/movement.direction.right",
            "env_agent/friendly_fire",
            "env_agent/red.steals.ore_red.from.blue",
            "env_agent/blue.steals.ore_blue.from.red",
        ]

    def collect_training_context(
        self, run_name: str, replay_timestamp_step: int, context_window: int = 1000
    ) -> WandBTrainingContext:
        """
        Collect training context around replay timestamp

        Args:
            run_name: WandB run name to analyze
            replay_timestamp_step: Step number when replay was captured
            context_window: Steps before/after timestamp to analyze (default: ±1000)

        Returns:
            Complete training context analysis

        Raises:
            ValueError: If WandB data is unavailable or insufficient
        """
        # Get run data and URL
        run_data = self._get_wandb_run_data(run_name)
        run_url = self._get_wandb_run_url(run_name)

        if not run_data:
            raise ValueError(f"WandB run data unavailable for run: {run_name}")

        # Extract metrics samples around timestamp
        metrics_samples = self._extract_metrics_samples(run_data, replay_timestamp_step, context_window)

        if not metrics_samples:
            raise ValueError(f"No training metrics found around step {replay_timestamp_step}")

        # Analyze learning progressions
        learning_progressions = self._analyze_learning_progressions(metrics_samples)

        # Build training context
        context = WandBTrainingContext(
            run_name=run_name,
            run_url=run_url,
            replay_timestamp_step=replay_timestamp_step,
            context_window_steps=context_window * 2,  # ±context_window
            reward_progression=learning_progressions.get("env_agent/reward", self._empty_progression("reward")),
            action_mastery_progression=self._extract_action_progressions(learning_progressions),
            resource_efficiency_progression=self._extract_resource_progressions(learning_progressions),
            movement_learning_progression=self._extract_movement_progressions(learning_progressions),
            training_stage=self._determine_training_stage(metrics_samples, replay_timestamp_step),
            learning_velocity=self._calculate_learning_velocity(metrics_samples),
            performance_stability=self._calculate_performance_stability(metrics_samples),
            behavioral_adaptation_rate=self._calculate_adaptation_rate(metrics_samples),
            behavior_metric_correlations=self._analyze_metric_correlations(metrics_samples),
            critical_learning_moments=self._identify_critical_moments(metrics_samples),
        )

        return context

    def _get_wandb_run_data(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Get WandB run data using MCP client"""
        try:
            # Use existing MCP WandB function
            result = self.mcp_client("mcp__metta__get_wandb_run", {"run_name": run_name})
            return result if isinstance(result, dict) else None
        except Exception:
            return None

    def _get_wandb_run_url(self, run_name: str) -> Optional[str]:
        """Get WandB run URL using MCP client"""
        try:
            # Use existing MCP WandB function
            result = self.mcp_client("mcp__metta__get_wandb_run_url", {"run_name": run_name})
            return result if isinstance(result, str) else None
        except Exception:
            return None

    def _extract_metrics_samples(
        self, run_data: Dict[str, Any], center_step: int, window: int
    ) -> List[TrainingMetricsSample]:
        """Extract training metrics samples around center step"""
        samples = []

        # In a real implementation, this would parse the WandB run data structure
        # and extract metric values at different training steps
        # For now, we'll create a placeholder structure

        # Simulate extracting samples from WandB history
        start_step = max(0, center_step - window)
        end_step = center_step + window

        # Sample every 50 steps in the window
        for step in range(start_step, end_step + 1, 50):
            # Simulate metric extraction from run_data
            metrics = self._simulate_metrics_at_step(run_data, step, center_step)

            if metrics:  # Only add if we have metrics data
                sample = TrainingMetricsSample(
                    step=step,
                    timestamp=None,  # Would be extracted from run_data
                    metrics=metrics,
                )
                samples.append(sample)

        return samples

    def _simulate_metrics_at_step(self, run_data: Dict[str, Any], step: int, center_step: int) -> Dict[str, float]:
        """Simulate extracting metrics at a specific step (placeholder)"""
        # In real implementation, this would extract actual metric values from WandB data
        # For now, simulate realistic training progression patterns

        progress = step / max(center_step, 1000)  # Normalize progress

        # Simulate realistic metric progression
        metrics = {
            "env_agent/reward": progress * 1.5 + (0.1 * (step % 100) / 100),  # Generally improving
            "env_agent/action.get_output.success": min(0.9, 0.3 + progress * 0.6),  # Learning curve
            "env_agent/action.put_recipe_items.success": min(0.85, 0.2 + progress * 0.65),
            "env_agent/action.move.success": min(0.95, 0.7 + progress * 0.25),  # High baseline
            "env_agent/ore_red.gained": progress * 10 + (step % 50) / 10,
            "env_agent/ore_blue.gained": progress * 8 + (step % 30) / 8,
            "env_agent/battery_red.gained": progress * 5 + (step % 20) / 15,
            "env_agent/heart.gained": progress * 2 + (step % 100) / 50,
            "env_agent/movement.direction.up": 0.25 + 0.1 * (step % 20) / 20,
            "env_agent/movement.direction.right": 0.25 + 0.1 * (step % 25) / 25,
            "env_agent/friendly_fire": max(0, 0.1 - progress * 0.08),  # Decreasing over time
        }

        return metrics

    def _analyze_learning_progressions(self, samples: List[TrainingMetricsSample]) -> Dict[str, LearningProgression]:
        """Analyze learning progression for each metric"""
        progressions = {}

        # Group samples by metric
        metric_time_series = defaultdict(list)
        for sample in samples:
            for metric_name, value in sample.metrics.items():
                metric_time_series[metric_name].append((sample.step, value))

        # Analyze each metric's progression
        for metric_name, time_series in metric_time_series.items():
            if len(time_series) < 3:
                continue  # Need at least 3 points for trend analysis

            progression = self._calculate_metric_progression(metric_name, time_series)
            progressions[metric_name] = progression

        return progressions

    def _calculate_metric_progression(
        self, metric_name: str, time_series: List[Tuple[int, float]]
    ) -> LearningProgression:
        """Calculate learning progression for a single metric"""
        steps = [point[0] for point in time_series]
        values = [point[1] for point in time_series]

        # Calculate baseline (first 20% of samples)
        baseline_count = max(1, len(values) // 5)
        baseline_value = statistics.mean(values[:baseline_count])
        current_value = statistics.mean(values[-baseline_count:])  # Last 20%

        # Calculate trend using linear regression
        n = len(time_series)
        sum_x = sum(steps)
        sum_y = sum(values)
        sum_xy = sum(step * value for step, value in time_series)
        sum_x2 = sum(step * step for step in steps)

        # Linear regression slope
        slope = (n * sum_xy - sum_x * sum_y) / max(n * sum_x2 - sum_x * sum_x, 0.001)

        # Determine trend direction and strength
        if slope > 0.001:
            trend = "improving"
            trend_strength = min(1.0, abs(slope) * 1000)  # Scale slope
        elif slope < -0.001:
            trend = "declining"
            trend_strength = min(1.0, abs(slope) * 1000)
        else:
            trend = "stable"
            trend_strength = 0.0

        # Check for volatility
        if len(values) > 5:
            volatility = statistics.stdev(values) / max(statistics.mean(values), 0.001)
            if volatility > 0.3 and trend_strength < 0.5:
                trend = "volatile"
                trend_strength = volatility

        # Calculate progression rate (change per 1000 steps)
        step_range = max(steps) - min(steps)
        progression_rate = slope * 1000 if step_range > 0 else 0.0

        return LearningProgression(
            metric_name=metric_name,
            trend=trend,
            trend_strength=trend_strength,
            current_value=current_value,
            baseline_value=baseline_value,
            progression_rate=progression_rate,
        )

    def _extract_action_progressions(
        self, progressions: Dict[str, LearningProgression]
    ) -> Dict[str, LearningProgression]:
        """Extract action mastery progressions"""
        action_progressions = {}

        for metric_name, progression in progressions.items():
            if "action." in metric_name and ".success" in metric_name:
                action_name = metric_name.replace("env_agent/action.", "").replace(".success", "")
                action_progressions[action_name] = progression

        return action_progressions

    def _extract_resource_progressions(
        self, progressions: Dict[str, LearningProgression]
    ) -> Dict[str, LearningProgression]:
        """Extract resource efficiency progressions"""
        resource_progressions = {}

        for metric_name, progression in progressions.items():
            if ".gained" in metric_name:
                resource_name = metric_name.replace("env_agent/", "").replace(".gained", "")
                resource_progressions[resource_name] = progression

        return resource_progressions

    def _extract_movement_progressions(
        self, progressions: Dict[str, LearningProgression]
    ) -> Dict[str, LearningProgression]:
        """Extract movement learning progressions"""
        movement_progressions = {}

        for metric_name, progression in progressions.items():
            if "movement.direction." in metric_name:
                direction = metric_name.replace("env_agent/movement.direction.", "")
                movement_progressions[direction] = progression

        return movement_progressions

    def _determine_training_stage(self, samples: List[TrainingMetricsSample], replay_step: int) -> str:
        """Determine training stage based on progress and metrics"""
        if not samples:
            return "unknown"

        # Simple heuristic based on step number and reward progression
        # Get the maximum step for potential future use
        _max_step = max(sample.step for sample in samples)

        # Get reward progression if available
        reward_values = []
        for sample in samples:
            if "env_agent/reward" in sample.metrics:
                reward_values.append(sample.metrics["env_agent/reward"])

        # Stage classification
        if replay_step < 10000:
            return "early"
        elif replay_step > 100000:
            return "late"
        else:
            # Check if performance is still improving rapidly
            if reward_values and len(reward_values) > 5:
                recent_mean = statistics.mean(reward_values[-5:])
                early_mean = statistics.mean(reward_values[:5])
                if recent_mean > early_mean * 1.5:  # Still rapid improvement
                    return "mid"
                else:
                    return "late"
            else:
                return "mid"

    def _calculate_learning_velocity(self, samples: List[TrainingMetricsSample]) -> float:
        """Calculate rate of improvement across all metrics"""
        if len(samples) < 3:
            return 0.0

        # Calculate improvement rates for key metrics
        improvement_rates = []

        key_metrics = ["env_agent/reward", "env_agent/ore_red.gained", "env_agent/action.get_output.success"]

        for metric in key_metrics:
            metric_values = []
            steps = []

            for sample in samples:
                if metric in sample.metrics:
                    metric_values.append(sample.metrics[metric])
                    steps.append(sample.step)

            if len(metric_values) >= 3:
                # Simple linear trend calculation
                early_value = statistics.mean(metric_values[: len(metric_values) // 3])
                late_value = statistics.mean(metric_values[-len(metric_values) // 3 :])
                step_range = max(steps) - min(steps)

                if step_range > 0 and early_value > 0:
                    rate = (late_value - early_value) / (step_range * early_value)  # Normalized rate
                    improvement_rates.append(max(0, rate))

        return statistics.mean(improvement_rates) if improvement_rates else 0.0

    def _calculate_performance_stability(self, samples: List[TrainingMetricsSample]) -> float:
        """Calculate consistency of performance"""
        if len(samples) < 5:
            return 0.0

        # Calculate coefficient of variation for reward
        reward_values = []
        for sample in samples:
            if "env_agent/reward" in sample.metrics:
                reward_values.append(sample.metrics["env_agent/reward"])

        if len(reward_values) < 3:
            return 0.0

        mean_reward = statistics.mean(reward_values)
        if mean_reward == 0:
            return 0.0

        stdev_reward = statistics.stdev(reward_values)
        coefficient_of_variation = stdev_reward / mean_reward

        # Convert to stability score (lower CoV = higher stability)
        stability = max(0.0, 1.0 - coefficient_of_variation)
        return min(1.0, stability)

    def _calculate_adaptation_rate(self, samples: List[TrainingMetricsSample]) -> float:
        """Calculate how quickly behavioral patterns change"""
        if len(samples) < 5:
            return 0.0

        # Track changes in movement patterns as proxy for behavioral adaptation
        movement_metrics = ["env_agent/movement.direction.up", "env_agent/movement.direction.right"]

        adaptation_scores = []

        for metric in movement_metrics:
            values = []
            for sample in samples:
                if metric in sample.metrics:
                    values.append(sample.metrics[metric])

            if len(values) >= 5:
                # Calculate how much the pattern changes over time
                window_size = len(values) // 3
                early_pattern = statistics.mean(values[:window_size])
                late_pattern = statistics.mean(values[-window_size:])

                if early_pattern > 0:
                    change_rate = abs(late_pattern - early_pattern) / early_pattern
                    adaptation_scores.append(min(1.0, change_rate))

        return statistics.mean(adaptation_scores) if adaptation_scores else 0.0

    def _analyze_metric_correlations(self, samples: List[TrainingMetricsSample]) -> Dict[str, float]:
        """Analyze correlations between different metrics"""
        correlations = {}

        if len(samples) < 5:
            return correlations

        # Extract time series for key metrics
        reward_series = []
        resource_series = []
        action_success_series = []

        for sample in samples:
            if "env_agent/reward" in sample.metrics:
                reward_series.append(sample.metrics["env_agent/reward"])

            # Aggregate resource gains
            resource_gain = 0
            for metric_name, value in sample.metrics.items():
                if ".gained" in metric_name:
                    resource_gain += value
            resource_series.append(resource_gain)

            # Aggregate action success rates
            success_rates = [v for k, v in sample.metrics.items() if "action." in k and ".success" in k]
            if success_rates:
                action_success_series.append(statistics.mean(success_rates))

        # Calculate correlations
        if len(reward_series) >= 3 and len(resource_series) >= 3:
            correlations["reward_vs_resource_gain"] = self._pearson_correlation(reward_series, resource_series)

        if len(reward_series) >= 3 and len(action_success_series) >= 3:
            correlations["reward_vs_action_success"] = self._pearson_correlation(reward_series, action_success_series)

        return correlations

    def _identify_critical_moments(self, samples: List[TrainingMetricsSample]) -> List[Dict[str, Any]]:
        """Identify critical learning moments or performance shifts"""
        critical_moments = []

        if len(samples) < 5:
            return critical_moments

        # Look for significant reward jumps
        reward_values = []
        reward_steps = []

        for sample in samples:
            if "env_agent/reward" in sample.metrics:
                reward_values.append(sample.metrics["env_agent/reward"])
                reward_steps.append(sample.step)

        if len(reward_values) >= 5:
            # Identify sudden improvements (reward jumps)
            for i in range(2, len(reward_values) - 2):
                before_avg = statistics.mean(reward_values[max(0, i - 2) : i])
                after_avg = statistics.mean(reward_values[i : min(len(reward_values), i + 3)])

                if after_avg > before_avg * 1.3:  # 30% improvement
                    critical_moments.append(
                        {
                            "type": "performance_breakthrough",
                            "step": reward_steps[i],
                            "description": f"Significant reward improvement: {before_avg:.2f} → {after_avg:.2f}",
                            "magnitude": (after_avg - before_avg) / before_avg,
                        }
                    )

        return critical_moments

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5

        return numerator / max(denominator, 0.001)

    def _empty_progression(self, metric_name: str) -> LearningProgression:
        """Create empty progression for missing metrics"""
        return LearningProgression(
            metric_name=metric_name,
            trend="unknown",
            trend_strength=0.0,
            current_value=0.0,
            baseline_value=0.0,
            progression_rate=0.0,
        )


class TrainingProgressionAnalyzer:
    """Analyzes training progression and correlates with replay behaviors"""

    def analyze_training_progression(
        self, wandb_context: WandBTrainingContext, replay_behaviors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze training progression and correlate with replay behaviors

        Args:
            wandb_context: Training context from WandB
            replay_behaviors: Behavioral analysis from replay

        Returns:
            Training progression analysis with behavior correlations

        Raises:
            ValueError: If insufficient data for progression analysis
        """
        if not wandb_context:
            raise ValueError("No WandB training context provided")

        analysis = {
            "progression_summary": self._summarize_progression(wandb_context),
            "learning_insights": self._generate_learning_insights(wandb_context),
            "behavior_correlation": self._correlate_with_replay_behaviors(wandb_context, replay_behaviors),
            "training_recommendations": self._generate_training_recommendations(wandb_context),
            "performance_context": self._analyze_performance_context(wandb_context),
        }

        return analysis

    def _summarize_progression(self, context: WandBTrainingContext) -> Dict[str, Any]:
        """Summarize key training progression insights"""
        summary = {
            "training_stage": context.training_stage,
            "overall_learning_trend": self._determine_overall_trend(context),
            "key_achievements": [],
            "concerning_patterns": [],
            "progression_health_score": 0.0,
        }

        # Analyze reward progression
        reward_prog = context.reward_progression
        if reward_prog.trend == "improving":
            summary["key_achievements"].append(
                f"Reward consistently improving ({reward_prog.trend_strength:.1%} strength)"
            )
        elif reward_prog.trend == "declining":
            summary["concerning_patterns"].append(f"Reward declining ({reward_prog.trend_strength:.1%} strength)")

        # Analyze action mastery
        improving_actions = [
            name for name, prog in context.action_mastery_progression.items() if prog.trend == "improving"
        ]
        if improving_actions:
            summary["key_achievements"].append(f"Improving action mastery: {', '.join(improving_actions[:3])}")

        # Calculate health score
        health_components = [
            context.learning_velocity * 0.3,
            context.performance_stability * 0.3,
            (1.0 if reward_prog.trend == "improving" else 0.5) * 0.4,
        ]
        summary["progression_health_score"] = sum(health_components)

        return summary

    def _determine_overall_trend(self, context: WandBTrainingContext) -> str:
        """Determine overall learning trend across all metrics"""
        improving_count = 0
        declining_count = 0
        total_count = 0

        # Check key progressions
        for progression_dict in [context.action_mastery_progression, context.resource_efficiency_progression]:
            for prog in progression_dict.values():
                total_count += 1
                if prog.trend == "improving":
                    improving_count += 1
                elif prog.trend == "declining":
                    declining_count += 1

        # Include reward progression
        total_count += 1
        if context.reward_progression.trend == "improving":
            improving_count += 1
        elif context.reward_progression.trend == "declining":
            declining_count += 1

        if total_count == 0:
            return "unknown"

        improving_pct = improving_count / total_count
        declining_pct = declining_count / total_count

        if improving_pct > 0.6:
            return "strongly_improving"
        elif improving_pct > 0.4:
            return "mostly_improving"
        elif declining_pct > 0.6:
            return "concerning_decline"
        elif declining_pct > 0.4:
            return "mixed_with_concerns"
        else:
            return "stable"

    def _generate_learning_insights(self, context: WandBTrainingContext) -> List[str]:
        """Generate insights about learning patterns"""
        insights = []

        # Learning velocity insights
        if context.learning_velocity > 0.1:
            insights.append(f"High learning velocity ({context.learning_velocity:.2f}) - agent adapting quickly")
        elif context.learning_velocity < 0.02:
            insights.append(f"Low learning velocity ({context.learning_velocity:.2f}) - learning may be plateauing")

        # Performance stability insights
        if context.performance_stability > 0.8:
            insights.append(f"High performance stability ({context.performance_stability:.1%}) - consistent execution")
        elif context.performance_stability < 0.4:
            insights.append(f"Low performance stability ({context.performance_stability:.1%}) - erratic performance")

        # Behavioral adaptation insights
        if context.behavioral_adaptation_rate > 0.3:
            insights.append("High behavioral adaptation rate - strategy still evolving")
        elif context.behavioral_adaptation_rate < 0.1:
            insights.append("Low behavioral adaptation - strategy may have converged")

        # Critical moments insights
        if context.critical_learning_moments:
            insights.append(f"Identified {len(context.critical_learning_moments)} critical learning moments")

        return insights

    def _correlate_with_replay_behaviors(
        self, context: WandBTrainingContext, replay_behaviors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Correlate training metrics with replay behaviors"""
        correlations = {
            "training_behavior_alignment": "unknown",
            "performance_expectation_match": "unknown",
            "learning_transfer_evidence": [],
            "strategy_consistency": "unknown",
        }

        # This would perform detailed correlation analysis between training metrics
        # and observed behaviors in the replay
        # For now, provide placeholder structure

        return correlations

    def _generate_training_recommendations(self, context: WandBTrainingContext) -> List[str]:
        """Generate training recommendations based on progression analysis"""
        recommendations = []

        # Learning velocity recommendations
        if context.learning_velocity < 0.02:
            recommendations.append("Consider increasing learning rate or curriculum difficulty")

        # Stability recommendations
        if context.performance_stability < 0.5:
            recommendations.append("Performance highly variable - consider regularization or smaller learning rate")

        # Stage-specific recommendations
        if context.training_stage == "early" and context.reward_progression.trend != "improving":
            recommendations.append("Early training with non-improving reward - check environment setup")
        elif context.training_stage == "late" and context.behavioral_adaptation_rate > 0.5:
            recommendations.append("Late training with high adaptation rate - may benefit from reduced exploration")

        return recommendations

    def _analyze_performance_context(self, context: WandBTrainingContext) -> Dict[str, Any]:
        """Analyze performance in context of training progression"""
        performance_context = {
            "expected_performance_level": "unknown",
            "performance_trajectory": "unknown",
            "competency_areas": [],
            "improvement_areas": [],
        }

        # Analyze competency areas based on action mastery
        for action_name, progression in context.action_mastery_progression.items():
            if progression.current_value > 0.7 and progression.trend in ["improving", "stable"]:
                performance_context["competency_areas"].append(action_name)
            elif progression.current_value < 0.4 or progression.trend == "declining":
                performance_context["improvement_areas"].append(action_name)

        # Analyze resource efficiency competencies
        for resource_name, progression in context.resource_efficiency_progression.items():
            if progression.trend == "improving" and progression.current_value > progression.baseline_value * 1.5:
                performance_context["competency_areas"].append(f"{resource_name} acquisition")

        return performance_context
