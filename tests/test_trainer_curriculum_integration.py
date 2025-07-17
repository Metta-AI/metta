#!/usr/bin/env python3
"""Test trainer's curriculum stats integration."""

from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.rl.curriculum_client import CurriculumClient
from metta.rl.curriculum_server import CurriculumServer


class TestCurriculum(Curriculum):
    """Test curriculum with stats."""
    
    def __init__(self):
        self.task_count = 0
        self.completed_tasks = []
        
    def get_task(self) -> Task:
        self.task_count += 1
        task_id = f"task_{self.task_count % 3}"
        env_cfg = OmegaConf.create({
            "game": {
                "width": 10,
                "height": 10,
                "num_agents": 2,
                "max_steps": 100
            }
        })
        return Task(task_id, self, env_cfg)
    
    def complete_task(self, id: str, score: float):
        self.completed_tasks.append((id, score))
        
    def get_env_cfg_by_bucket(self) -> dict[str, DictConfig]:
        return {
            "task_0": OmegaConf.create({"game": {"width": 10, "height": 10}}),
            "task_1": OmegaConf.create({"game": {"width": 11, "height": 11}}),
            "task_2": OmegaConf.create({"game": {"width": 12, "height": 12}})
        }
    
    def get_completion_rates(self) -> dict[str, float]:
        return {
            "task_completions/task_0": 0.3,
            "task_completions/task_1": 0.4,
            "task_completions/task_2": 0.3
        }
    
    def get_task_probs(self) -> dict[str, float]:
        return {
            "task_0": 0.25,
            "task_1": 0.35,
            "task_2": 0.40
        }
    
    def get_curriculum_stats(self) -> dict:
        return {
            "total_tasks": self.task_count,
            "completed_tasks": len(self.completed_tasks),
            "learning_rate": 0.75,
            "difficulty": 2.5
        }


def test_trainer_stats_collection_with_curriculum():
    """Test that trainer correctly collects and logs curriculum stats."""
    # Import here to avoid circular imports
    from metta.rl.functions import build_wandb_stats, compute_timing_stats, process_training_stats
    
    # Create mock trainer components
    trainer_cfg = MagicMock()
    trainer_cfg.kickstart = MagicMock()
    
    # Create curriculum
    curriculum = TestCurriculum()
    
    # Simulate some task completions
    for i in range(5):
        task = curriculum.get_task()
        curriculum.complete_task(task.id(), 0.7 + i * 0.05)
    
    # Mock stats and components
    raw_stats = {"reward": [0.5, 0.6, 0.7], "episode_length": [100, 110, 120]}
    losses = MagicMock()
    losses.minibatches_processed = 10
    losses.policy_loss_sum = 0.5
    losses.value_loss_sum = 0.3
    losses.entropy_sum = 0.1
    losses.explained_variance = 0.85
    losses.approx_kl_sum = 0.02
    losses.clipfrac_sum = 0.1
    
    experience = MagicMock()
    experience.num_minibatches = 4
    
    kickstarter = MagicMock()
    kickstarter.enabled = False
    
    timer = MagicMock()
    timer.get_elapsed.return_value = 100.0
    timer.get_last_elapsed.return_value = 10.0
    
    # Process training stats
    processed_stats = process_training_stats(
        raw_stats=raw_stats,
        losses=losses,
        experience=experience,
        trainer_config=trainer_cfg,
        kickstarter=kickstarter
    )
    
    # Compute timing stats
    timing_info = compute_timing_stats(timer=timer, agent_step=1000)
    
    # Build parameters
    parameters = {
        "learning_rate": 0.001,
        "epoch_steps": 100,
        "num_minibatches": 4,
        "generation": 1,
        "latest_saved_policy_epoch": 10
    }
    
    # Collect curriculum stats as done in trainer._process_stats()
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
    
    # Build complete stats
    all_stats = build_wandb_stats(
        processed_stats=processed_stats,
        timing_info=timing_info,
        weight_stats={},
        grad_stats={},
        system_stats={},
        memory_stats={},
        parameters=parameters,
        evals=MagicMock(),
        agent_step=1000,
        epoch=10
    )
    
    # Add curriculum stats
    all_stats.update(curriculum_stats)
    
    # Verify curriculum stats are present
    assert "curriculum/total_tasks" in all_stats
    assert "curriculum/completed_tasks" in all_stats
    assert "curriculum/learning_rate" in all_stats
    assert "curriculum/difficulty" in all_stats
    
    assert "curriculum/task_prob/task_0" in all_stats
    assert "curriculum/task_prob/task_1" in all_stats
    assert "curriculum/task_prob/task_2" in all_stats
    
    assert "task_completions/task_0" in all_stats
    assert "task_completions/task_1" in all_stats
    assert "task_completions/task_2" in all_stats
    
    # Verify values
    assert all_stats["curriculum/total_tasks"] >= 5
    assert all_stats["curriculum/completed_tasks"] == 5
    assert all_stats["curriculum/learning_rate"] == 0.75
    assert all_stats["curriculum/difficulty"] == 2.5
    
    assert abs(all_stats["curriculum/task_prob/task_0"] - 0.25) < 0.01
    assert abs(all_stats["curriculum/task_prob/task_1"] - 0.35) < 0.01
    assert abs(all_stats["curriculum/task_prob/task_2"] - 0.40) < 0.01
    
    assert abs(all_stats["task_completions/task_0"] - 0.3) < 0.01
    assert abs(all_stats["task_completions/task_1"] - 0.4) < 0.01
    assert abs(all_stats["task_completions/task_2"] - 0.3) < 0.01


def test_trainer_with_curriculum_server_client():
    """Test trainer integration with curriculum server and client."""
    # Start a curriculum server
    curriculum = TestCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15561)
    server.start(background=True)
    
    try:
        # Create client
        client = CurriculumClient(
            server_url="http://127.0.0.1:15561",
            batch_size=10
        )
        
        # Simulate what trainer does
        # 1. Get tasks
        tasks = []
        for _ in range(5):
            task = client.get_task()
            assert task is not None
            tasks.append(task)
        
        # 2. Try to get stats from client (should return empty)
        assert client.get_curriculum_stats() == {}
        assert client.get_task_probs() == {}
        assert client.get_completion_rates() == {}
        
        # 3. Complete task (no-op on client)
        client.complete_task("task_0", 0.8)
        
        # 4. Server curriculum should still have its stats
        server_stats = curriculum.get_curriculum_stats()
        assert server_stats["total_tasks"] >= 5
        
        # 5. Test env config by bucket
        configs = client.get_env_cfg_by_bucket()
        assert len(configs) > 0
        
        # Each config should have game settings
        for name, cfg in configs.items():
            assert "game" in cfg
            assert "width" in cfg.game
            assert "height" in cfg.game
            
    finally:
        server.stop()


def test_curriculum_client_no_server():
    """Test client behavior when server is not available."""
    # Try to create client with non-existent server
    client = CurriculumClient(
        server_url="http://localhost:19999",
        batch_size=10,
        max_retries=1,
        retry_delay=0.1
    )
    
    # Should raise error when trying to get task
    with pytest.raises(RuntimeError) as exc_info:
        client.get_task()
    
    assert "Failed to fetch tasks" in str(exc_info.value)


def test_curriculum_server_concurrent_requests():
    """Test server handling concurrent requests."""
    import concurrent.futures
    import time
    
    curriculum = TestCurriculum()
    server = CurriculumServer(curriculum, host="127.0.0.1", port=15562)
    server.start(background=True)
    time.sleep(0.5)
    
    try:
        def fetch_tasks(client_id):
            client = CurriculumClient(
                server_url="http://127.0.0.1:15562",
                batch_size=5
            )
            tasks = []
            for _ in range(3):
                task = client.get_task()
                tasks.append((client_id, task.name))
            return tasks
        
        # Run multiple clients concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(fetch_tasks, i)
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
        
        # Verify all clients got tasks
        assert len(all_results) == 15  # 5 clients * 3 tasks each
        
        # All tasks should be valid
        for client_id, task_name in all_results:
            assert task_name in ["task_0", "task_1", "task_2"]
            
    finally:
        server.stop()


if __name__ == "__main__":
    test_trainer_stats_collection_with_curriculum()
    test_trainer_with_curriculum_server_client()
    test_curriculum_client_no_server()
    test_curriculum_server_concurrent_requests()
    print("All trainer integration tests passed!")