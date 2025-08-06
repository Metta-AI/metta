"""
Tests for curriculum analysis functionality.

This module tests the curriculum regret analysis framework and ensures
it integrates properly with the existing curriculum testing infrastructure.
"""

import numpy as np
from omegaconf import OmegaConf

from metta.eval.curriculum_analysis import (
    CurriculumRegretAnalyzer,
    CurriculumScenarioAnalyzer,
    create_curriculum_metrics,
)
from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum
from metta.mettagrid.curriculum.prioritize_regressed import PrioritizeRegressedCurriculum
from metta.mettagrid.curriculum.random import RandomCurriculum


class TestCurriculumAnalysis:
    """Test the curriculum analysis functionality."""

    def test_curriculum_metrics_creation(self):
        """Test creation of CurriculumMetrics objects."""
        task_weights = {"task_a": 0.5, "task_b": 0.5}
        sampling_history = [{"task_a": 0.6, "task_b": 0.4}, {"task_a": 0.4, "task_b": 0.6}]

        metrics = create_curriculum_metrics(
            efficiency=100.0,
            time_to_threshold=50,
            time_to_first_mastery=10,
            final_perf_variance=0.1,
            task_weights=task_weights,
            sampling_history=sampling_history,
        )

        assert metrics.efficiency == 100.0
        assert metrics.time_to_threshold == 50
        assert metrics.time_to_first_mastery == 10
        assert metrics.final_perf_variance == 0.1
        assert metrics.task_weights == task_weights
        assert metrics.sampling_history == sampling_history

    def test_regret_calculation_basic(self):
        """Test basic regret calculation."""
        analyzer = CurriculumRegretAnalyzer(max_epochs=200)

        # Create test metrics
        oracle_metrics = create_curriculum_metrics(
            efficiency=150.0,
            time_to_threshold=30,
            time_to_first_mastery=5,
            final_perf_variance=0.05,
            task_weights={"task_a": 0.5, "task_b": 0.5},
        )

        curriculum_metrics = create_curriculum_metrics(
            efficiency=100.0,
            time_to_threshold=50,
            time_to_first_mastery=10,
            final_perf_variance=0.1,
            task_weights={"task_a": 0.6, "task_b": 0.4},
        )

        regret = analyzer.calculate_regret(curriculum_metrics, oracle_metrics)

        assert regret.efficiency_regret == 50.0  # 150 - 100
        assert regret.time_regret == 20  # 50 - 30
        assert regret.normalized_efficiency_regret == 50.0 / 150.0
        assert regret.normalized_time_regret == 20.0 / 200.0

    def test_regret_calculation_edge_cases(self):
        """Test regret calculation with edge cases."""
        analyzer = CurriculumRegretAnalyzer(max_epochs=200)

        oracle_metrics = create_curriculum_metrics(
            efficiency=100.0,
            time_to_threshold=50,
            time_to_first_mastery=10,
            final_perf_variance=0.1,
            task_weights={"task_a": 0.5, "task_b": 0.5},
        )

        # Test curriculum failure
        failed_curriculum = create_curriculum_metrics(
            efficiency=50.0,
            time_to_threshold=-1,  # Failed
            time_to_first_mastery=10,
            final_perf_variance=0.1,
            task_weights={"task_a": 0.5, "task_b": 0.5},
        )

        regret = analyzer.calculate_regret(failed_curriculum, oracle_metrics)
        assert regret.time_regret == 200  # Max penalty for failure

        # Test curriculum success when oracle fails
        successful_curriculum = create_curriculum_metrics(
            efficiency=150.0,
            time_to_threshold=30,
            time_to_first_mastery=5,
            final_perf_variance=0.05,
            task_weights={"task_a": 0.5, "task_b": 0.5},
        )

        failed_oracle = create_curriculum_metrics(
            efficiency=50.0,
            time_to_threshold=-1,  # Failed
            time_to_first_mastery=10,
            final_perf_variance=0.1,
            task_weights={"task_a": 0.5, "task_b": 0.5},
        )

        regret = analyzer.calculate_regret(successful_curriculum, failed_oracle)
        assert regret.time_regret == -200  # Negative regret (better than oracle)

    def test_curriculum_comparison(self):
        """Test curriculum comparison functionality."""
        analyzer = CurriculumRegretAnalyzer(max_epochs=200)

        # Create test curricula results
        curricula_results = {
            "oracle": create_curriculum_metrics(
                efficiency=150.0,
                time_to_threshold=30,
                time_to_first_mastery=5,
                final_perf_variance=0.05,
                task_weights={"task_a": 0.5, "task_b": 0.5},
            ),
            "random": create_curriculum_metrics(
                efficiency=100.0,
                time_to_threshold=50,
                time_to_first_mastery=10,
                final_perf_variance=0.1,
                task_weights={"task_a": 0.6, "task_b": 0.4},
            ),
            "lp": create_curriculum_metrics(
                efficiency=120.0,
                time_to_threshold=40,
                time_to_first_mastery=8,
                final_perf_variance=0.08,
                task_weights={"task_a": 0.55, "task_b": 0.45},
            ),
        }

        comparison_df = analyzer.compare_curricula(curricula_results)

        assert len(comparison_df) == 3
        assert "efficiency_regret" in comparison_df.columns
        assert "time_regret" in comparison_df.columns

        # Oracle should have zero regret
        oracle_row = comparison_df[comparison_df["curriculum"] == "oracle"].iloc[0]
        assert oracle_row["efficiency_regret"] == 0.0
        assert oracle_row["time_regret"] == 0

    def test_adaptation_analysis(self):
        """Test curriculum adaptation analysis."""
        analyzer = CurriculumRegretAnalyzer(max_epochs=200)

        # Create metrics with sampling history
        sampling_history = [
            {"task_a": 0.6, "task_b": 0.4},
            {"task_a": 0.4, "task_b": 0.6},
            {"task_a": 0.5, "task_b": 0.5},
            {"task_a": 0.7, "task_b": 0.3},
        ]

        metrics = create_curriculum_metrics(
            efficiency=100.0,
            time_to_threshold=50,
            time_to_first_mastery=10,
            final_perf_variance=0.1,
            task_weights={"task_a": 0.6, "task_b": 0.4},
            sampling_history=sampling_history,
        )

        adaptation = analyzer.analyze_curriculum_adaptation(metrics)

        assert "adaptation_speed" in adaptation
        assert "weight_stability" in adaptation
        assert adaptation["adaptation_speed"] >= 0.0
        assert adaptation["weight_stability"] >= 0.0
        assert adaptation["weight_stability"] <= 1.0

    def test_scenario_analyzer(self):
        """Test scenario analyzer functionality."""
        regret_analyzer = CurriculumRegretAnalyzer(max_epochs=200)
        scenario_analyzer = CurriculumScenarioAnalyzer(regret_analyzer)

        # Create scenario results
        scenario_results = {
            "scenario_1": {
                "oracle": create_curriculum_metrics(
                    efficiency=150.0,
                    time_to_threshold=30,
                    time_to_first_mastery=5,
                    final_perf_variance=0.05,
                    task_weights={"task_a": 0.5, "task_b": 0.5},
                ),
                "random": create_curriculum_metrics(
                    efficiency=100.0,
                    time_to_threshold=50,
                    time_to_first_mastery=10,
                    final_perf_variance=0.1,
                    task_weights={"task_a": 0.6, "task_b": 0.4},
                ),
            },
            "scenario_2": {
                "oracle": create_curriculum_metrics(
                    efficiency=120.0,
                    time_to_threshold=40,
                    time_to_first_mastery=8,
                    final_perf_variance=0.08,
                    task_weights={"task_a": 0.55, "task_b": 0.45},
                ),
                "random": create_curriculum_metrics(
                    efficiency=80.0,
                    time_to_threshold=60,
                    time_to_first_mastery=15,
                    final_perf_variance=0.12,
                    task_weights={"task_a": 0.7, "task_b": 0.3},
                ),
            },
        }

        comparison_df = scenario_analyzer.run_scenario_comparison(scenario_results)

        assert len(comparison_df) == 4  # 2 scenarios × 2 curricula
        assert "scenario" in comparison_df.columns
        assert "adaptation_speed" in comparison_df.columns
        assert "weight_stability" in comparison_df.columns

        # Test summary report
        summary = scenario_analyzer.generate_summary_report(comparison_df)

        assert summary["total_scenarios"] == 2
        assert summary["total_curricula"] == 1  # Excluding oracle
        assert "curriculum_rankings" in summary
        assert "random" in summary["curriculum_rankings"]

    def test_integration_with_existing_curricula(self):
        """Test integration with existing curriculum classes."""
        # This test ensures the analysis framework works with actual curriculum instances
        # even though we can't run full simulations in unit tests

        # Create curriculum instances
        curricula = {
            "random": RandomCurriculum({"task_a": 1.0, "task_b": 1.0}, OmegaConf.create({})),
            "lp": LearningProgressCurriculum({"task_a": 1.0, "task_b": 1.0}, OmegaConf.create({})),
            "regressed": PrioritizeRegressedCurriculum({"task_a": 1.0, "task_b": 1.0}, OmegaConf.create({})),
        }

        # Create mock metrics for these curricula
        curricula_results = {}
        for name, curriculum in curricula.items():
            # Mock metrics - in real usage these would come from simulation
            curricula_results[name] = create_curriculum_metrics(
                efficiency=100.0 + np.random.rand() * 50,  # Random efficiency
                time_to_threshold=30 + int(np.random.rand() * 30),
                time_to_first_mastery=5 + int(np.random.rand() * 10),
                final_perf_variance=0.05 + np.random.rand() * 0.1,
                task_weights=curriculum.get_task_probs()
                if hasattr(curriculum, "get_task_probs")
                else {"task_a": 0.5, "task_b": 0.5},
            )

        # Add oracle
        curricula_results["oracle"] = create_curriculum_metrics(
            efficiency=150.0,
            time_to_threshold=25,
            time_to_first_mastery=3,
            final_perf_variance=0.02,
            task_weights={"task_a": 0.5, "task_b": 0.5},
        )

        # Test analysis
        analyzer = CurriculumRegretAnalyzer(max_epochs=200)
        comparison_df = analyzer.compare_curricula(curricula_results)

        assert len(comparison_df) == 4  # 4 curricula including oracle
        assert all(curriculum in comparison_df["curriculum"].values for curriculum in curricula_results.keys())
