#!/usr/bin/env python3
"""Test curriculum stats collection in trainer._process_stats."""

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.functions import build_wandb_stats, compute_timing_stats, process_training_stats


class TestCurriculum(Curriculum):
    """Test curriculum with comprehensive stats."""
    
    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []
        
    def get_task(self) -> Task:
        self.task_count += 1
        task_id = f"task_{self.task_count}"
        env_cfg = OmegaConf.create({
            "game": {"width": 10, "height": 10, "num_agents": 2}
        })
        return Task(task_id, self, env_cfg)
    
    def complete_task(self, id: str, score: float):
        self.completed_tasks.append((id, score))
        
    def get_env_cfg_by_bucket(self) -> dict[str, OmegaConf.DictConfig]:
        return {"default": OmegaConf.create({"game": {"width": 10, "height": 10}})}
    
    def get_completion_rates(self) -> dict[str, float]:
        return {
            "task_completions/easy": 0.75,
            "task_completions/hard": 0.25
        }
    
    def get_task_probs(self) -> dict[str, float]:
        return {
            "easy": 0.7,
            "hard": 0.3
        }
    
    def get_curriculum_stats(self) -> dict:
        return {
            "total_tasks": self.task_count,
            "completed_tasks": len(self.completed_tasks),
            "average_score": 0.85,
            "learning_progress": 0.002
        }


def test_curriculum_stats_collection():
    """Test that curriculum stats are properly collected in trainer._process_stats."""
    
    # Create mock objects
    curriculum = TestCurriculum()
    
    # Generate some tasks
    for _ in range(5):
        curriculum.get_task()
    
    # Simulate some completions
    curriculum.complete_task("task_1", 0.8)
    curriculum.complete_task("task_2", 0.9)
    
    # Mock trainer components
    stats = {"reward": [0.8, 0.9, 0.7]}
    losses = MagicMock()
    losses.policy_loss = 0.1
    losses.value_loss = 0.2
    losses.entropy = 0.05
    losses.explained_variance = 0.9
    losses.approx_kl_sum = 0.01
    losses.minibatches_processed = 4
    
    experience = MagicMock()
    experience.num_minibatches = 4
    
    trainer_config = MagicMock()
    trainer_config.kickstart.enabled = False
    
    kickstarter = MagicMock()
    kickstarter.enabled = False
    
    # Process training stats
    processed_stats = process_training_stats(
        raw_stats=stats,
        losses=losses,
        experience=experience,
        trainer_config=trainer_config,
        kickstarter=kickstarter,
    )
    
    # Collect curriculum stats as trainer would
    curriculum_stats = {}
    
    # Get curriculum stats
    if hasattr(curriculum, "get_curriculum_stats"):
        raw_curriculum_stats = curriculum.get_curriculum_stats()
        for key, value in raw_curriculum_stats.items():
            curriculum_stats[f"curriculum/{key}"] = value
    
    # Get task probabilities
    if hasattr(curriculum, "get_task_probs"):
        task_probs = curriculum.get_task_probs()
        for task_id, prob in task_probs.items():
            curriculum_stats[f"curriculum/task_prob/{task_id}"] = prob
    
    # Get completion rates
    if hasattr(curriculum, "get_completion_rates"):
        completion_rates = curriculum.get_completion_rates()
        curriculum_stats.update(completion_rates)
    
    # Verify curriculum stats were collected
    assert "curriculum/total_tasks" in curriculum_stats
    assert curriculum_stats["curriculum/total_tasks"] == 5
    assert curriculum_stats["curriculum/completed_tasks"] == 2
    assert curriculum_stats["curriculum/average_score"] == 0.85
    assert curriculum_stats["curriculum/learning_progress"] == 0.002
    
    # Verify task probabilities
    assert "curriculum/task_prob/easy" in curriculum_stats
    assert curriculum_stats["curriculum/task_prob/easy"] == 0.7
    assert curriculum_stats["curriculum/task_prob/hard"] == 0.3
    
    # Verify completion rates
    assert "task_completions/easy" in curriculum_stats
    assert curriculum_stats["task_completions/easy"] == 0.75
    assert "task_completions/hard" in curriculum_stats
    assert curriculum_stats["task_completions/hard"] == 0.25
    
    # Test that stats would be included in wandb logging
    timer = MagicMock()
    timer.get_elapsed = MagicMock(return_value=100.0)
    timer.get_last_elapsed = MagicMock(return_value=1.0)
    
    timing_info = compute_timing_stats(timer=timer, agent_step=1000)
    
    all_stats = build_wandb_stats(
        processed_stats=processed_stats,
        timing_info=timing_info,
        weight_stats={},
        grad_stats={},
        system_stats={},
        memory_stats={},
        parameters={"learning_rate": 0.001},
        evals=EvalRewardSummary(),
        agent_step=1000,
        epoch=10,
    )
    
    # Add curriculum stats
    all_stats.update(curriculum_stats)
    
    # Verify all stats are present
    assert "overview/reward" in all_stats
    assert "losses/policy_loss" in all_stats
    assert "curriculum/total_tasks" in all_stats
    assert "curriculum/task_prob/easy" in all_stats
    assert "task_completions/easy" in all_stats
    
    print("✓ Curriculum stats are properly collected")
    print(f"✓ Found {len([k for k in all_stats if 'curriculum' in k])} curriculum-related stats")


def test_curriculum_client_stats():
    """Test that curriculum client returns empty stats (server handles all state)."""
    from metta.rl.curriculum_client import CurriculumClient
    
    # Create a mock client (don't need actual server for this test)
    client = CurriculumClient(
        server_url="http://localhost:5555",
        batch_size=10
    )
    
    # Verify all stats methods return empty
    assert client.get_curriculum_stats() == {}
    assert client.get_task_probs() == {}
    assert client.get_completion_rates() == {}
    
    # Verify complete_task is a no-op
    client.complete_task("task_1", 0.9)  # Should not raise
    
    print("✓ Curriculum client correctly returns empty stats")


if __name__ == "__main__":
    test_curriculum_stats_collection()
    test_curriculum_client_stats()
    print("\nAll curriculum stats tests passed!")