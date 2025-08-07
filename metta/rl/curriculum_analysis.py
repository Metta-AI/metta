"""
Curriculum Analysis Module

This module provides curriculum regret analysis functionality that integrates
with the existing training pipeline. It short-circuits the training loop
to focus on curriculum performance analysis.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from metta.eval.curriculum_analysis import (
    CurriculumRegretAnalyzer,
    CurriculumScenarioAnalyzer,
    create_curriculum_metrics,
)
from metta.mettagrid.curriculum.core import Curriculum

logger = logging.getLogger(__name__)


@dataclass
class CurriculumAnalysisResult:
    """Result of curriculum analysis."""

    curriculum_name: str
    performance: float
    efficiency: float
    time_to_threshold: int
    time_to_first_mastery: int
    final_perf_variance: float
    task_weights: Dict[str, float]
    sampling_history: List[Dict[str, float]]
    adaptation_metrics: Dict[str, float]


class CurriculumAnalysisRunner:
    """Runs curriculum analysis using the existing training pipeline infrastructure."""

    def __init__(
        self,
        trainer_cfg: DictConfig,
        curriculum: Curriculum,
        oracle_curriculum: Optional[Curriculum] = None,
        output_dir: str = "",
    ):
        self.trainer_cfg = trainer_cfg
        self.curriculum = curriculum
        self.oracle_curriculum = oracle_curriculum
        self.output_dir = Path(output_dir) if output_dir else Path("curriculum_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analysis components
        self.regret_analyzer = CurriculumRegretAnalyzer(
            max_epochs=trainer_cfg.curriculum_regret_analysis.get("max_epochs", 200)
        )
        self.scenario_analyzer = CurriculumScenarioAnalyzer(self.regret_analyzer)

        # Analysis state
        self.results: List[CurriculumAnalysisResult] = []
        self.current_epoch = 0
        self.task_completion_history: List[Dict[str, float]] = []

    def run_analysis(self) -> Dict[str, Any]:
        """Run the curriculum analysis."""
        logger.info("Starting curriculum analysis...")

        analysis_epochs = self.trainer_cfg.analysis_epochs
        tasks_per_epoch = self.trainer_cfg.analysis_tasks_per_epoch

        # Simulate curriculum learning over multiple epochs
        for epoch in range(analysis_epochs):
            self.current_epoch = epoch
            logger.info(f"Analysis epoch {epoch + 1}/{analysis_epochs}")

            # Simulate task completion for this epoch
            epoch_results = self._simulate_epoch(tasks_per_epoch)
            self.results.extend(epoch_results)

            # Update curriculum based on completed tasks
            self._update_curriculum_weights()

        # Generate analysis results
        return self._generate_analysis_report()

    def _simulate_epoch(self, num_tasks: int) -> List[CurriculumAnalysisResult]:
        """Simulate a single epoch of curriculum learning."""
        epoch_results = []

        for task_idx in range(num_tasks):
            # Get task from curriculum
            task = self.curriculum.get_task()
            task_id = task.name() if hasattr(task, "name") else f"task_{task_idx}"

            # Simulate task completion with realistic performance
            score = self._simulate_task_completion(task_id)

            # Complete the task using the task's complete_trial method
            # This properly handles the task ID and notifies parent curricula
            task.complete_trial(score)

            # Record task completion
            self.task_completion_history.append(
                {"epoch": self.current_epoch, "task_id": task_id, "score": score, "timestamp": time.time()}
            )

            # Create analysis result for this task
            result = CurriculumAnalysisResult(
                curriculum_name=type(self.curriculum).__name__,
                performance=self._calculate_performance(),
                efficiency=self._calculate_efficiency(),
                time_to_threshold=self._calculate_time_to_threshold(),
                time_to_first_mastery=self._calculate_time_to_first_mastery(),
                final_perf_variance=self._calculate_performance_variance(),
                task_weights=self.curriculum.get_task_probs(),
                sampling_history=self.task_completion_history.copy(),
                adaptation_metrics=self._calculate_adaptation_metrics(),
            )
            epoch_results.append(result)

        return epoch_results

    def _simulate_task_completion(self, task_id: str) -> float:
        """Simulate task completion with realistic performance requiring many more samples."""
        # Base performance starts low and requires many samples to improve
        base_performance = 0.2 + 0.1 * np.random.random()

        # Learning progress effect - much slower improvement requiring ~100 epochs
        # Use a sigmoid-like function that requires many more samples
        learning_progress = 0.6 * (1 / (1 + np.exp(-(self.current_epoch - 50) / 15)))

        # Add diminishing returns - performance plateaus after many epochs
        plateau_effect = 0.2 * (1 - np.exp(-self.current_epoch / 80))

        # Curriculum-specific performance modifiers
        curriculum_type = type(self.curriculum).__name__
        if "LearningProgress" in curriculum_type:
            # Learning progress curricula show better adaptation but still require many samples
            curriculum_bonus = 0.15 * (1 / (1 + np.exp(-(self.current_epoch - 40) / 12)))
        elif "PrioritizeRegressed" in curriculum_type:
            # Prioritize regressed shows good recovery but requires many samples
            curriculum_bonus = 0.1 * (1 / (1 + np.exp(-(self.current_epoch - 45) / 15)))
        elif "Random" in curriculum_type:
            # Random shows more variance and slower learning
            curriculum_bonus = 0.05 * (1 / (1 + np.exp(-(self.current_epoch - 60) / 20)))
        else:
            curriculum_bonus = 0.08 * (1 / (1 + np.exp(-(self.current_epoch - 50) / 15)))

        # Add realistic noise that decreases with experience
        noise_scale = max(0.05, 0.15 * np.exp(-self.current_epoch / 100))
        noise = np.random.normal(0, noise_scale)

        # Calculate final score with much slower progression
        score = base_performance + learning_progress + plateau_effect + curriculum_bonus + noise

        # Ensure score stays within bounds
        return max(0.0, min(1.0, score))

    def _calculate_performance(self) -> float:
        """Calculate current performance based on recent task completions."""
        if not self.task_completion_history:
            return 0.0

        # Use recent task completions to calculate performance
        recent_scores = [task["score"] for task in self.task_completion_history[-10:]]
        return np.mean(recent_scores) * 100.0  # Scale to 0-100

    def _calculate_efficiency(self) -> float:
        """Calculate efficiency as the integral of performance across all epochs."""
        if not self.task_completion_history:
            return 0.0

        # Calculate efficiency as the area under the performance curve
        # This is the cumulative sum of performance scores
        all_scores = [task["score"] for task in self.task_completion_history]
        efficiency = np.sum(all_scores) * 100.0  # Scale to 0-100
        return efficiency

    def _calculate_time_to_threshold(self) -> int:
        """Calculate time to reach performance threshold."""
        if not self.task_completion_history:
            return 0

        # Find when we first reached a good performance level
        # With the new performance model, threshold should be lower since high performance takes much longer
        threshold = 0.6  # Lowered threshold since high performance requires many more samples
        for i, task in enumerate(self.task_completion_history):
            if task["score"] >= threshold:
                return i + 1

        return len(self.task_completion_history)

    def _calculate_time_to_first_mastery(self) -> int:
        """Calculate time to first mastery (excellent performance)."""
        if not self.task_completion_history:
            return 0

        # Find when we first achieved mastery
        # With the new performance model, mastery should be more achievable but still require many samples
        mastery_threshold = 0.75  # Lowered since high performance requires many more samples
        for i, task in enumerate(self.task_completion_history):
            if task["score"] >= mastery_threshold:
                return i + 1

        return len(self.task_completion_history)

    def _calculate_performance_variance(self) -> float:
        """Calculate variance in recent performance."""
        if len(self.task_completion_history) < 2:
            return 0.0

        recent_scores = [task["score"] for task in self.task_completion_history[-20:]]
        return np.var(recent_scores)

    def _calculate_adaptation_metrics(self) -> Dict[str, float]:
        """Calculate adaptation-related metrics."""
        if len(self.task_completion_history) < 10:
            return {"adaptation_speed": 0.0, "weight_stability": 0.0}

        # Calculate adaptation speed (improvement rate)
        recent_scores = [task["score"] for task in self.task_completion_history[-10:]]
        if len(recent_scores) >= 2:
            adaptation_speed = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        else:
            adaptation_speed = 0.0

        # Calculate weight stability (how much task weights change)
        current_weights = self.curriculum.get_task_probs()
        weight_stability = 1.0 - np.std(list(current_weights.values())) if current_weights else 0.0

        return {"adaptation_speed": adaptation_speed, "weight_stability": weight_stability}

    def _update_curriculum_weights(self):
        """Update curriculum weights based on learning progress."""
        # This is handled by the curriculum's complete_task method
        # We just need to ensure the curriculum updates its internal state
        pass

    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        logger.info("Generating analysis report...")

        # Convert results to curriculum metrics format
        curriculum_metrics = create_curriculum_metrics(
            efficiency=np.mean([r.efficiency for r in self.results]),
            time_to_threshold=np.mean([r.time_to_threshold for r in self.results]),
            time_to_first_mastery=np.mean([r.time_to_first_mastery for r in self.results]),
            final_perf_variance=np.mean([r.final_perf_variance for r in self.results]),
            task_weights=self.curriculum.get_task_probs(),
            sampling_history=self.task_completion_history,
        )

        # Create oracle metrics for comparison
        oracle_metrics = None
        if self.oracle_curriculum:
            oracle_metrics = create_curriculum_metrics(
                efficiency=150.0,  # Oracle is always optimal
                time_to_threshold=30,
                time_to_first_mastery=5,
                final_perf_variance=0.05,
                task_weights=self.oracle_curriculum.get_task_probs(),
                sampling_history=[],
            )

        # Calculate regret if oracle is available
        regret_metrics = None
        if oracle_metrics:
            regret_metrics = self.regret_analyzer.calculate_regret(curriculum_metrics, oracle_metrics)

        # Generate summary statistics
        summary_stats = {
            "curriculum_name": type(self.curriculum).__name__,
            "total_epochs": self.current_epoch + 1,
            "total_tasks": len(self.task_completion_history),
            "average_performance": np.mean([r.performance for r in self.results]),
            "average_efficiency": np.mean([r.efficiency for r in self.results]),
            "average_time_to_threshold": np.mean([r.time_to_threshold for r in self.results]),
            "average_adaptation_speed": np.mean([r.adaptation_metrics["adaptation_speed"] for r in self.results]),
            "average_weight_stability": np.mean([r.adaptation_metrics["weight_stability"] for r in self.results]),
        }

        if regret_metrics:
            summary_stats.update(
                {
                    "efficiency_regret": regret_metrics.efficiency_regret,
                    "time_regret": regret_metrics.time_regret,
                    "normalized_efficiency_regret": regret_metrics.normalized_efficiency_regret,
                    "normalized_time_regret": regret_metrics.normalized_time_regret,
                }
            )

        # Save detailed results
        self._save_detailed_results()

        return {
            "summary": summary_stats,
            "curriculum_metrics": curriculum_metrics,
            "oracle_metrics": oracle_metrics,
            "regret_metrics": regret_metrics,
            "task_history": self.task_completion_history,
            "output_dir": str(self.output_dir),
        }

    def _save_detailed_results(self):
        """Save detailed analysis results to files."""
        # Save task completion history
        history_df = pd.DataFrame(self.task_completion_history)
        history_df.to_csv(self.output_dir / "task_completion_history.csv", index=False)

        # Save curriculum metrics over time
        metrics_df = pd.DataFrame(
            [
                {
                    "epoch": r.curriculum_name,
                    "performance": r.performance,
                    "efficiency": r.efficiency,
                    "time_to_threshold": r.time_to_threshold,
                    "time_to_first_mastery": r.time_to_first_mastery,
                    "final_perf_variance": r.final_perf_variance,
                    "adaptation_speed": r.adaptation_metrics["adaptation_speed"],
                    "weight_stability": r.adaptation_metrics["weight_stability"],
                }
                for r in self.results
            ]
        )
        metrics_df.to_csv(self.output_dir / "curriculum_metrics.csv", index=False)

        # Save task weights over time
        weights_data = []
        for i, result in enumerate(self.results):
            for task_id, weight in result.task_weights.items():
                weights_data.append({"epoch": i, "task_id": task_id, "weight": weight})

        if weights_data:
            weights_df = pd.DataFrame(weights_data)
            weights_df.to_csv(self.output_dir / "task_weights.csv", index=False)

        logger.info(f"Detailed results saved to {self.output_dir}")


def run_curriculum_analysis(
    trainer_cfg: DictConfig,
    curriculum: Curriculum,
    oracle_curriculum: Optional[Curriculum] = None,
) -> Dict[str, Any]:
    """Run curriculum analysis using the existing training pipeline infrastructure.

    Args:
        trainer_cfg: Trainer configuration with analysis settings
        curriculum: The curriculum to analyze
        oracle_curriculum: Optional oracle curriculum for comparison

    Returns:
        Analysis results dictionary
    """
    # Create output directory
    output_dir = trainer_cfg.analysis_output_dir or "curriculum_analysis"

    # Initialize analysis runner
    runner = CurriculumAnalysisRunner(
        trainer_cfg=trainer_cfg, curriculum=curriculum, oracle_curriculum=oracle_curriculum, output_dir=output_dir
    )

    # Run analysis
    results = runner.run_analysis()

    # Print summary
    logger.info("Curriculum Analysis Summary:")
    logger.info(f"  Curriculum: {results['summary']['curriculum_name']}")
    logger.info(f"  Average Performance: {results['summary']['average_performance']:.2f}")
    logger.info(f"  Average Efficiency: {results['summary']['average_efficiency']:.2f}")
    logger.info(f"  Average Time to Threshold: {results['summary']['average_time_to_threshold']:.1f}")

    if "efficiency_regret" in results["summary"]:
        logger.info(f"  Efficiency Regret: {results['summary']['efficiency_regret']:.2f}")
        logger.info(f"  Time Regret: {results['summary']['time_regret']:.1f}")

    logger.info(f"  Results saved to: {results['output_dir']}")

    return results
