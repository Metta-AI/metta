#!/usr/bin/env python3
"""
Test script for AWS pricing integration with SkyPilot.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from metta.eval.aws_pricing import (
    AWSPricingClient,
    SkyPilotInstanceInfo,
    calculate_total_cost,
)
from metta.eval.performance_threshold_tracker import (
    PerformanceThreshold,
    PerformanceThresholdTracker,
)


def test_aws_pricing_client():
    """Test AWS pricing client functionality."""
    print("=== Testing AWS Pricing Client ===")

    # Test with different instance types
    test_cases = [
        ("g5.4xlarge", False),  # On-demand
        ("g5.4xlarge", True),  # Spot
        ("g5.8xlarge", False),  # On-demand
        ("g5.8xlarge", True),  # Spot
    ]

    pricing_client = AWSPricingClient()

    for instance_type, use_spot in test_cases:
        price = pricing_client.get_instance_pricing(instance_type, use_spot)
        spot_text = "spot" if use_spot else "on-demand"
        print(f"{instance_type} ({spot_text}): ${price:.3f}/hour")

    print()


def test_skypilot_instance_info():
    """Test SkyPilot instance information extraction."""
    print("=== Testing SkyPilot Instance Info ===")

    # Test environment variable detection
    instance_type, use_spot, num_nodes, num_gpus = SkyPilotInstanceInfo.get_instance_info()
    print("Environment detection:")
    print(f"  Instance type: {instance_type}")
    print(f"  Use spot: {use_spot}")
    print(f"  Num nodes: {num_nodes}")
    print(f"  GPUs per node: {num_gpus}")
    print()

    # Test task YAML parsing
    task_yaml_path = "devops/skypilot/config/sk_train.yaml"
    if os.path.exists(task_yaml_path):
        instance_type, use_spot, num_nodes, num_gpus = SkyPilotInstanceInfo.get_instance_info_from_task(task_yaml_path)
        print("Task YAML parsing:")
        print(f"  Instance type: {instance_type}")
        print(f"  Use spot: {use_spot}")
        print(f"  Num nodes: {num_nodes}")
        print(f"  GPUs per node: {num_gpus}")
        print()


def test_cost_calculation():
    """Test cost calculation with different scenarios."""
    print("=== Testing Cost Calculation ===")

    # Test scenarios
    scenarios = [
        {
            "hours": 2.5,
            "instance_type": "g5.4xlarge",
            "use_spot": False,
            "num_nodes": 1,
            "num_gpus_per_node": 1,
        },
        {
            "hours": 4.0,
            "instance_type": "g5.4xlarge",
            "use_spot": True,
            "num_nodes": 1,
            "num_gpus_per_node": 1,
        },
        {
            "hours": 8.0,
            "instance_type": "g5.8xlarge",
            "use_spot": False,
            "num_nodes": 2,
            "num_gpus_per_node": 1,
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        cost = calculate_total_cost(**scenario)
        print(
            f"Scenario {i}: {scenario['hours']}h × {scenario['instance_type']} "
            f"({'spot' if scenario['use_spot'] else 'on-demand'}) "
            f"× {scenario['num_nodes']} nodes = ${cost:.2f}"
        )
    print()


def test_performance_threshold_tracker():
    """Test performance threshold tracker with new pricing system."""
    print("=== Testing Performance Threshold Tracker ===")

    # Create test thresholds
    thresholds = [
        PerformanceThreshold(
            name="heart_gained_2",
            metric="env_agent/heart.gained",
            target_value=2.0,
            smoothing_factor=0.3,  # Higher smoothing factor for faster convergence
        ),
        PerformanceThreshold(
            name="heart_gained_5",
            metric="env_agent/heart.gained",
            target_value=5.0,
            smoothing_factor=0.3,  # Higher smoothing factor for faster convergence
        ),
    ]

    # Available metrics
    available_metrics = ["env_agent/heart.gained", "env_agent/reward"]

    # Create tracker
    tracker = PerformanceThresholdTracker(thresholds, available_metrics)

    # Simulate training progression
    print("Simulating training progression...")

    # Simulate metric progression with more aggressive values
    heart_gained_values = [0.5, 1.0, 1.5, 2.5, 3.5, 4.5, 6.0, 7.0]

    for i, heart_gained in enumerate(heart_gained_values):
        # Simulate time progression (2 hours total)
        elapsed_time = (i + 1) * 2 * 3600 / len(heart_gained_values)  # 2 hours total

        metrics = {
            "env_agent/heart.gained": heart_gained,
            "env_agent/reward": heart_gained * 10,
        }

        samples = (i + 1) * 1000  # Simulate sample progression

        # Update tracker
        tracker.update(
            metrics=metrics,
            samples=samples,
            elapsed_time=elapsed_time,
            instance_type="g5.4xlarge",
            use_spot=True,
            num_nodes=1,
            num_gpus_per_node=1,
        )

        print(
            f"Step {i + 1}: heart.gained = {heart_gained:.1f}, samples = {samples}, time = {elapsed_time / 3600:.1f}h"
        )

        # Check if thresholds reached
        results = tracker.get_results()
        for threshold_name, result in results.items():
            if result.achieved:
                print(
                    f"  ✓ {threshold_name} reached: "
                    f"{result.samples_to_threshold} samples, "
                    f"{result.minutes_to_threshold:.1f} min, "
                    f"${result.cost_to_threshold:.2f}"
                )

    print()

    # Show final results
    print("Final results:")
    results = tracker.get_results()
    for threshold_name, result in results.items():
        if result.achieved:
            print(
                f"  {threshold_name}: Achieved at {result.samples_to_threshold} samples, "
                f"{result.minutes_to_threshold:.1f} minutes, ${result.cost_to_threshold:.2f}"
            )
        else:
            print(f"  {threshold_name}: Not achieved")

    print()


def main():
    """Run all tests."""
    print("Testing AWS Pricing Integration with SkyPilot")
    print("=" * 50)
    print()

    try:
        test_aws_pricing_client()
        test_skypilot_instance_info()
        test_cost_calculation()
        test_performance_threshold_tracker()

        print("All tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
