#!/usr/bin/env -S uv run
"""Example usage of the progressive forgetting curriculum.

This script demonstrates how to:
1. Create a progressive forgetting curriculum
2. Run training with the curriculum
3. Analyze the results
"""

import logging

from metta.curriculum.analysis import ForgettingAnalyzer
from metta.curriculum.progressive_forgetting import ProgressiveForgettingCurriculum


def main():
    """Run a simple progressive forgetting experiment."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Define task sets for the experiment
    task_sets = {
        "navigation": {
            "/env/mettagrid/navigation/evals/emptyspace_withinsight": 1,
            "/env/mettagrid/navigation/evals/obstacles1": 1,
        },
        "memory": {
            "/env/mettagrid/memory/evals/easy": 1,
            "/env/mettagrid/memory/evals/medium": 1,
        },
    }

    logger.info("Creating progressive forgetting curriculum...")

    # Create the curriculum
    curriculum = ProgressiveForgettingCurriculum(
        task_sets=task_sets,
        performance_threshold=0.8,
        smoothing=0.1,
        switch_interval=100,  # Short for demo
        eval_interval=10,  # Short for demo
        randomize_order=False,
    )

    logger.info(f"Curriculum initialized with task sets: {list(task_sets.keys())}")
    logger.info(f"Current task set: {curriculum.current_task_set}")

    # Simulate some training steps
    logger.info("Simulating training steps...")

    # Train on navigation tasks first
    for step in range(50):
        # Simulate high performance on navigation
        curriculum.complete_task("/env/mettagrid/navigation/evals/emptyspace_withinsight", 0.9)
        curriculum.complete_task("/env/mettagrid/navigation/evals/obstacles1", 0.85)

        # Log progress
        if step % 10 == 0:
            stats = curriculum.get_curriculum_stats()
            logger.info(f"Step {step}: {stats}")

    # Continue training (should switch to memory)
    for step in range(50, 100):
        # Simulate mixed performance
        if curriculum.current_task_set == "navigation":
            curriculum.complete_task("/env/mettagrid/navigation/evals/emptyspace_withinsight", 0.9)
            curriculum.complete_task("/env/mettagrid/navigation/evals/obstacles1", 0.85)
        else:
            curriculum.complete_task("/env/mettagrid/memory/evals/easy", 0.7)
            curriculum.complete_task("/env/mettagrid/memory/evals/medium", 0.6)

        # Log progress
        if step % 10 == 0:
            stats = curriculum.get_curriculum_stats()
            logger.info(f"Step {step}: {stats}")

    logger.info("Training simulation completed!")

    # Show final stats
    final_stats = curriculum.get_curriculum_stats()
    logger.info("Final curriculum stats:")
    for key, value in final_stats.items():
        logger.info(f"  {key}: {value}")

    # Demonstrate analysis (this would normally use real training data)
    logger.info("\nDemonstrating analysis capabilities...")

    # Create a dummy analyzer
    analyzer = ForgettingAnalyzer("dummy_path")

    # Create some dummy performance data
    dummy_performances = {
        "navigation": [(i, 0.8 + 0.1 * (i / 50)) for i in range(100)],
        "memory": [(i, 0.2 + 0.6 * ((i - 50) / 50) if i >= 50 else 0.2) for i in range(100)],
    }

    # Calculate forgetting metrics
    forgetting_metrics = analyzer.calculate_forgetting_metrics(dummy_performances)

    logger.info("Calculated forgetting metrics:")
    logger.info(f"Total pairs: {len(forgetting_metrics)}")
    for pair_name, metrics in forgetting_metrics.items():
        logger.info(f"  {pair_name}:")
        for metric_name, value in metrics.items():
            logger.info(f"    {metric_name}: {value:.4f}")

    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    main()
