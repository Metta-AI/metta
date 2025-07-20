#!/usr/bin/env -S uv run
"""
Generate comprehensive test data for the Metta dashboard application.

This script creates training runs, epochs, policies, and episodes to test:
- Training runs listing and details
- Description editing functionality
- Tag management and filtering
- Heatmap visualization with multiple suites/metrics
- Policy selector functionality
- Search functionality
- URL parameter persistence

Usage:
    python generate_test_data.py

Requirements:
    - Stats server running at 127.0.0.1:8000
    - Database initialized and accessible
"""

import uuid
from typing import Dict, List, Optional, TypedDict

import httpx

from metta.app_backend.stats_client import (
    ClientEpochResponse,
    ClientPolicyResponse,
    ClientTrainingRunResponse,
    StatsClient,
)


class EpochConfig(TypedDict):
    start: int
    end: int
    lr: str
    performance: str


class TrainingRunConfig(TypedDict):
    user: str
    token: str
    name: str
    description: str
    tags: List[str]
    url: Optional[str]
    algorithm: str
    env_type: str
    epochs: List[EpochConfig]


class TaskSuite(TypedDict):
    tasks: List[str]
    metrics: List[str]


class PolicyData(TypedDict):
    policy: ClientPolicyResponse
    epoch: ClientEpochResponse
    config: EpochConfig
    name: str


class CreatedRunData(TypedDict):
    run: ClientTrainingRunResponse
    config: TrainingRunConfig


def create_machine_token(base_url: str, user_email: str, token_name: str) -> str:
    """Create a machine token for the given user."""
    with httpx.Client(base_url=base_url) as client:
        response = client.post(
            "/tokens",
            json={"name": token_name},
            headers={"X-Auth-Request-Email": user_email},
        )
        response.raise_for_status()
        return response.json()["token"]


def update_training_run_metadata(
    base_url: str, token: str, run_id: uuid.UUID, description: str, tags: List[str]
) -> None:
    """Update training run description and tags."""
    with httpx.Client(base_url=base_url) as client:
        headers = {"X-Auth-Token": token}

        # Update description
        desc_response = client.put(
            f"/dashboard/training-runs/{run_id}/description",
            json={"description": description},
            headers=headers,
        )
        desc_response.raise_for_status()

        # Update tags
        tags_response = client.put(
            f"/dashboard/training-runs/{run_id}/tags",
            json={"tags": tags},
            headers=headers,
        )
        tags_response.raise_for_status()


def generate_test_data():
    """Generate comprehensive test data for the dashboard application."""
    base_url = "http://127.0.0.1:8000"

    print("🚀 Starting test data generation...")

    # Create machine tokens for different users
    print("📝 Creating machine tokens...")
    user1_token = create_machine_token(base_url, "alice@example.com", "test_data_generator_alice")
    user2_token = create_machine_token(base_url, "bob@example.com", "test_data_generator_bob")
    user3_token = create_machine_token(base_url, "charlie@example.com", "test_data_generator_charlie")

    # Training run configurations with rich metadata
    training_runs_config: List[TrainingRunConfig] = [
        # Alice's runs - Deep Learning experiments
        {
            "user": "alice@example.com",
            "token": user1_token,
            "name": "deep_rl_navigation_v1",
            "description": "Deep reinforcement learning experiment for navigation tasks using PPO with curriculum.",
            "tags": ["deep-learning", "navigation", "ppo", "curriculum", "baseline"],
            "url": "https://wandb.ai/alice/deep-rl-nav/runs/nav-v1",
            "algorithm": "PPO",
            "env_type": "navigation",
            "epochs": [
                {"start": 0, "end": 100, "lr": "1e-4", "performance": "early"},
                {"start": 101, "end": 300, "lr": "5e-5", "performance": "improving"},
                {"start": 301, "end": 500, "lr": "1e-5", "performance": "converged"},
            ],
        },
        {
            "user": "alice@example.com",
            "token": user1_token,
            "name": "multi_agent_cooperation",
            "description": "Multi-agent cooperation study with varying team sizes and communication protocols.",
            "tags": ["multi-agent", "cooperation", "communication", "emergence", "research"],
            "url": "https://wandb.ai/alice/multi-agent/runs/coop-v2",
            "algorithm": "MADDPG",
            "env_type": "cooperation",
            "epochs": [
                {"start": 0, "end": 150, "lr": "3e-4", "performance": "random"},
                {"start": 151, "end": 400, "lr": "1e-4", "performance": "learning"},
                {"start": 401, "end": 600, "lr": "5e-5", "performance": "cooperative"},
            ],
        },
        # Bob's runs - Optimization focus
        {
            "user": "bob@example.com",
            "token": user2_token,
            "name": "hyperparameter_optimization_study",
            "description": "Systematic hyperparameter optimization using Optuna for robotic manipulation tasks.",
            "tags": ["optimization", "manipulation", "optuna", "hyperparameters", "systematic"],
            "url": None,
            "algorithm": "SAC",
            "env_type": "manipulation",
            "epochs": [
                {"start": 0, "end": 80, "lr": "2e-4", "performance": "baseline"},
                {"start": 81, "end": 200, "lr": "1e-4", "performance": "optimized"},
            ],
        },
        {
            "user": "bob@example.com",
            "token": user2_token,
            "name": "curriculum_learning_experiment",
            "description": "Curriculum learning approach for complex navigation environments.",
            "tags": ["curriculum", "navigation", "progressive", "scheduling", "adaptive"],
            "url": "https://wandb.ai/bob/curriculum/runs/nav-curr-v1",
            "algorithm": "PPO",
            "env_type": "navigation",
            "epochs": [
                {"start": 0, "end": 120, "lr": "5e-4", "performance": "easy_tasks"},
                {"start": 121, "end": 280, "lr": "2e-4", "performance": "medium_tasks"},
                {"start": 281, "end": 450, "lr": "1e-4", "performance": "hard_tasks"},
            ],
        },
        # Charlie's runs - Safety and robustness
        {
            "user": "charlie@example.com",
            "token": user3_token,
            "name": "safety_constrained_rl",
            "description": "Safety-constrained reinforcement learning with cost functions and safe exploration.",
            "tags": ["safety", "constraints", "exploration", "cost-functions", "robustness"],
            "url": "https://wandb.ai/charlie/safety-rl/runs/safe-v1",
            "algorithm": "CPO",
            "env_type": "safety_critical",
            "epochs": [
                {"start": 0, "end": 100, "lr": "1e-4", "performance": "safe_random"},
                {"start": 101, "end": 250, "lr": "5e-5", "performance": "constrained_learning"},
            ],
        },
        {
            "user": "charlie@example.com",
            "token": user3_token,
            "name": "adversarial_robustness_test",
            "description": "Testing agent robustness against adversarial perturbations and domain shift.",
            "tags": ["robustness", "adversarial", "generalization", "domain-shift", "testing"],
            "url": None,
            "algorithm": "TRPO",
            "env_type": "adversarial",
            "epochs": [
                {"start": 0, "end": 200, "lr": "3e-4", "performance": "standard"},
                {"start": 201, "end": 350, "lr": "1e-4", "performance": "robust"},
            ],
        },
    ]

    # Evaluation suites and tasks for comprehensive testing
    eval_suites: Dict[str, TaskSuite] = {
        "navigation": {
            "tasks": ["maze_easy", "maze_hard", "obstacle_course", "multi_goal", "dynamic_obstacles"],
            "metrics": ["reward", "success_rate", "path_efficiency", "collision_count", "time_to_goal"],
        },
        "manipulation": {
            "tasks": ["pick_and_place", "door_opening", "object_stacking", "fine_motor", "tool_use"],
            "metrics": ["reward", "success_rate", "precision", "force_control", "task_completion_time"],
        },
        "cooperation": {
            "tasks": ["team_navigation", "resource_sharing", "communication_task", "coordination_challenge"],
            "metrics": ["reward", "team_success", "communication_efficiency", "coordination_score"],
        },
        "safety_critical": {
            "tasks": ["safe_navigation", "emergency_stop", "constraint_satisfaction", "risk_assessment"],
            "metrics": ["reward", "safety_violations", "constraint_satisfaction", "risk_score"],
        },
        "adversarial": {
            "tasks": ["perturbed_env", "domain_transfer", "noise_robustness", "attack_resistance"],
            "metrics": ["reward", "robustness_score", "adaptation_speed", "failure_rate"],
        },
    }

    print("🏃 Creating training runs and episodes...")

    created_runs: List[CreatedRunData] = []

    for run_config in training_runs_config:
        print(f"  📊 Creating training run: {run_config['name']} for {run_config['user']}")

        # Create StatsClient for this user
        with httpx.Client(base_url=base_url) as http_client:
            stats_client = StatsClient(http_client, run_config["token"])

            # Create training run
            training_run = stats_client.create_training_run(
                name=run_config["name"],
                attributes={
                    "algorithm": run_config["algorithm"],
                    "env_type": run_config["env_type"],
                    "experiment_type": "test_data",
                },
                url=run_config["url"],
            )

            # Update description and tags
            update_training_run_metadata(
                base_url, run_config["token"], training_run.id, run_config["description"], run_config["tags"]
            )

            created_runs.append({"run": training_run, "config": run_config})

            # Create epochs and policies for this run
            policies: List[PolicyData] = []
            for _i, epoch_config in enumerate(run_config["epochs"]):
                epoch = stats_client.create_epoch(
                    run_id=training_run.id,
                    start_training_epoch=epoch_config["start"],
                    end_training_epoch=epoch_config["end"],
                    attributes={"learning_rate": epoch_config["lr"], "performance_stage": epoch_config["performance"]},
                )

                # Create policy for this epoch
                policy_name = f"{run_config['name']}_epoch_{epoch_config['end']}"
                policy = stats_client.create_policy(
                    name=policy_name,
                    description=f"Policy after {epoch_config['end']} epochs - {epoch_config['performance']} stage",
                    url=f"https://storage.example.com/policies/{policy_name}.pt",
                    epoch_id=epoch.id,
                )
                policies.append({"policy": policy, "epoch": epoch, "config": epoch_config, "name": policy_name})

            # Create comprehensive episode data for relevant suites
            env_type = run_config["env_type"]
            relevant_suites = [env_type] if env_type in eval_suites else ["navigation"]

            # Add navigation suite for all runs to ensure cross-suite comparison
            if "navigation" not in relevant_suites:
                relevant_suites.append("navigation")

            for suite_name in relevant_suites:
                suite = eval_suites[suite_name]
                print(f"    🎯 Creating episodes for {suite_name} suite...")

                for task in suite["tasks"]:
                    for policy_data in policies:
                        policy = policy_data["policy"]
                        epoch = policy_data["epoch"]
                        stage = policy_data["config"]["performance"]

                        # Generate realistic performance based on training stage
                        performance_multipliers = {
                            "early": 0.3,
                            "random": 0.2,
                            "baseline": 0.4,
                            "improving": 0.6,
                            "learning": 0.5,
                            "optimized": 0.8,
                            "converged": 0.9,
                            "cooperative": 0.85,
                            "easy_tasks": 0.7,
                            "medium_tasks": 0.8,
                            "hard_tasks": 0.75,
                            "safe_random": 0.4,
                            "constrained_learning": 0.7,
                            "standard": 0.6,
                            "robust": 0.8,
                        }
                        base_performance = performance_multipliers.get(stage, 0.5)

                        # Add some task-specific variation
                        task_difficulty = {
                            "maze_easy": 1.2,
                            "maze_hard": 0.7,
                            "obstacle_course": 0.8,
                            "pick_and_place": 0.9,
                            "door_opening": 0.6,
                            "object_stacking": 0.5,
                            "team_navigation": 0.8,
                            "resource_sharing": 0.7,
                            "safe_navigation": 0.9,
                            "emergency_stop": 0.95,
                            "perturbed_env": 0.6,
                            "domain_transfer": 0.4,
                        }
                        difficulty_factor = task_difficulty.get(task, 0.8)

                        # Generate metrics for this episode
                        agent_metrics: Dict[int, Dict[str, float]] = {}
                        num_agents = 2 if suite_name == "cooperation" else 1

                        for agent_id in range(num_agents):
                            metrics = {}
                            for metric in suite["metrics"]:
                                if metric == "reward":
                                    base_reward = 100.0
                                    final_reward = base_reward * base_performance * difficulty_factor
                                    # Add some noise
                                    import random

                                    random.seed(hash((policy.id, task, metric, agent_id)) % (2**32))
                                    noise = random.uniform(0.8, 1.2)
                                    metrics[metric] = final_reward * noise

                                elif "success" in metric or "efficiency" in metric:
                                    base_rate = base_performance * difficulty_factor
                                    import random

                                    random.seed(hash((policy.id, task, metric, agent_id)) % (2**32))
                                    noise = random.uniform(0.9, 1.1)
                                    metrics[metric] = min(1.0, base_rate * noise)

                                elif "time" in metric or "count" in metric:
                                    # Lower is better for these metrics
                                    base_value = 50.0
                                    # Better performance = lower time/count
                                    performance_factor = 2.0 - base_performance
                                    import random

                                    random.seed(hash((policy.id, task, metric, agent_id)) % (2**32))
                                    noise = random.uniform(0.8, 1.2)
                                    metrics[metric] = base_value * performance_factor * noise

                                else:
                                    # Generic positive metric
                                    base_value = 0.8
                                    import random

                                    random.seed(hash((policy.id, task, metric, agent_id)) % (2**32))
                                    noise = random.uniform(0.9, 1.1)
                                    metrics[metric] = base_value * base_performance * noise

                            agent_metrics[agent_id] = metrics

                        # Create episode
                        stats_client.record_episode(
                            agent_policies={aid: policy.id for aid in range(num_agents)},
                            agent_metrics=agent_metrics,
                            primary_policy_id=policy.id,
                            stats_epoch=epoch.id,
                            eval_name=f"{suite_name}/{task}",
                            simulation_suite=suite_name,
                            replay_url=f"https://replays.example.com/{policy_data['name']}/{suite_name}_{task}.mp4",
                            attributes={
                                "agent_groups": {str(aid): 1 if aid == 0 else 2 for aid in range(num_agents)},
                                "task_difficulty": difficulty_factor,
                                "episode_seed": hash((policy.id, task)) % 10000,
                                "environment_config": f"{suite_name}_standard",
                            },
                        )

    print("✅ Test data generation completed!")
    print(f"📈 Created {len(created_runs)} training runs with comprehensive episode data")
    print("\n🎯 Generated data includes:")
    print("  • Multiple users (alice, bob, charlie) with different expertise areas")
    print("  • Rich descriptions and diverse tag combinations for filtering tests")
    print("  • Multiple evaluation suites (navigation, manipulation, cooperation, etc.)")
    print("  • Progressive training epochs showing performance improvement")
    print("  • Comprehensive metrics for heatmap visualization")
    print("  • Multi-agent episodes for group analysis")
    print("  • Realistic performance curves and task difficulties")
    print("\n🧪 You can now test:")
    print("  ✓ Training runs listing and search functionality")
    print("  ✓ Description editing (for runs owned by each user)")
    print("  ✓ Tag management and filtering combinations")
    print("  ✓ URL parameter persistence for tag filters")
    print("  ✓ Heatmap visualization across different suites/metrics")
    print("  ✓ Policy selector (latest/best) functionality")
    print("  ✓ Individual training run detail pages")
    print("  ✓ Multi-user ownership and authorization")
    print("  ✓ Cross-suite performance comparison")

    # Print some example filter combinations to test
    print("\n🔍 Example tag filter combinations to test:")
    print("  • 'deep-learning' + 'navigation' (Alice's navigation experiments)")
    print("  • 'optimization' + 'systematic' (Bob's optimization studies)")
    print("  • 'safety' + 'robustness' (Charlie's safety research)")
    print("  • 'curriculum' (Both Alice and Bob's curriculum experiments)")
    print("  • 'baseline' (Baseline experiments across users)")

    # Print example URLs for sharing
    print("\n🔗 Example shareable URLs to test:")
    print("  • http://127.0.0.1:3000/training-runs?tag_filters=deep-learning,navigation")
    print("  • http://127.0.0.1:3000/training-runs?tag_filters=optimization,hyperparameters")
    print("  • http://127.0.0.1:3000/training-runs?tag_filters=safety,constraints")


if __name__ == "__main__":
    try:
        generate_test_data()
    except Exception as e:
        print(f"❌ Error generating test data: {e}")
        print("🔧 Make sure the stats server is running at http://127.0.0.1:8000")
        raise
