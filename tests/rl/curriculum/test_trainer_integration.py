#!/usr/bin/env python3
"""Comprehensive trainer integration tests with curriculum system.

This module tests all aspects of trainer-curriculum integration:
- Trainer stats collection with curriculum
- Server-client workflow from trainer perspective
- Batch exhaustion and prefetching behavior
- Expected curriculum interface methods
- Learning progress simulation
- Task distribution and completion patterns
"""

import time
from unittest.mock import MagicMock

import pytest

from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.curriculum.curriculum_client import CurriculumClient
from metta.rl.curriculum.curriculum_server import CurriculumServer
from metta.rl.util.stats import build_wandb_stats, compute_timing_stats, process_training_stats

from .conftest import MockCurriculum, StatefulCurriculum


class TestTrainerCurriculumIntegration:
    """Test trainer integration with curriculum system."""

    def test_trainer_expected_methods(self, free_port):
        """Test that curriculum client implements all methods expected by trainer."""
        # Create and start server
        curriculum = MockCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start(background=True)
        time.sleep(0.5)

        try:
            # Create client as trainer would
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=10
            )

            # Test all methods trainer expects:
            
            # 1. get_task() - returns Task with env_cfg() and complete()
            task = client.get_task()
            assert task is not None
            assert hasattr(task, "env_cfg")
            assert hasattr(task, "complete")
            assert hasattr(task, "name")
            assert hasattr(task, "id")
            assert callable(task.env_cfg)
            assert callable(task.complete)

            # 2. Task env_cfg() returns proper config
            env_cfg = task.env_cfg()
            assert "game" in env_cfg
            assert "num_agents" in env_cfg.game
            assert "max_steps" in env_cfg.game

            # 3. Multiple get_task calls work efficiently
            tasks = [client.get_task() for _ in range(20)]
            assert all(t is not None for t in tasks)
            assert len(tasks) == 20

            # 4. complete_task() doesn't raise (no-op on client)
            client.complete_task("task_1", 0.9)

            # 5. stats() returns empty dict
            stats = client.stats()
            assert isinstance(stats, dict)
            assert stats == {}

            # 6. Optional methods also work
            if hasattr(client, "get_task_probs"):
                assert client.get_task_probs() == {}
            if hasattr(client, "get_completion_rates"):
                assert client.get_completion_rates() == {}

        finally:
            client.stop()
            server.stop()

    def test_trainer_stats_collection_with_curriculum(self):
        """Test that trainer correctly collects and processes curriculum stats."""
        # Create curriculum
        curriculum = StatefulCurriculum()

        # Generate some activity
        for i in range(10):
            task = curriculum.get_task()
            curriculum.complete_task(task.id(), 0.7 + i * 0.02)

        # Mock trainer components
        raw_stats = {
            "reward": [0.8, 0.9, 0.7],
            "task_reward/default": [0.8, 0.9, 0.7],
            "episode_length": [100, 110, 120]
        }
        
        losses = self._create_mock_losses()
        experience = self._create_mock_experience()
        trainer_config = self._create_mock_trainer_config()
        kickstarter = MagicMock(enabled=False)
        timer = self._create_mock_timer()

        # Process training stats
        processed_stats = process_training_stats(
            raw_stats=raw_stats,
            curriculum_stats=curriculum.stats(),
            losses=losses,
            experience=experience,
            trainer_config=trainer_config,
            kickstarter=kickstarter,
        )

        # Compute timing stats
        timing_info = compute_timing_stats(timer=timer, agent_step=1000)

        # Collect curriculum stats as trainer would
        curriculum_stats = {}
        
        # Get curriculum stats
        raw_curriculum_stats = curriculum.stats()
        for key, value in raw_curriculum_stats.items():
            curriculum_stats[f"curriculum/{key}"] = value

        # Get task probabilities
        task_probs = curriculum.get_task_probs()
        for task_id, prob in task_probs.items():
            curriculum_stats[f"curriculum/task_prob/{task_id}"] = prob

        # Get completion rates
        completion_rates = curriculum.get_completion_rates()
        curriculum_stats.update(completion_rates)

        # Build complete stats
        all_stats = build_wandb_stats(
            processed_stats=processed_stats,
            curriculum_stats=curriculum_stats,
            timing_info=timing_info,
            weight_stats={},
            grad_stats={},
            system_stats={},
            memory_stats={},
            parameters={"learning_rate": 0.001},
            hyperparameters={},
            evals=EvalRewardSummary(),
            agent_step=1000,
            epoch=10,
            world_size=1,
        )

        # Add curriculum stats
        all_stats.update(curriculum_stats)

        # Verify curriculum stats are present
        assert "curriculum/total_tasks" in all_stats
        assert all_stats["curriculum/total_tasks"] == 10
        assert all_stats["curriculum/completed_tasks"] == 10
        assert "curriculum/average_score" in all_stats
        assert "curriculum/learning_progress" in all_stats

        # Verify task probabilities
        assert "curriculum/task_prob/easy" in all_stats
        assert "curriculum/task_prob/hard" in all_stats

        # Verify completion rates
        assert "task_completions/easy" in all_stats
        assert "task_completions/hard" in all_stats

    def test_trainer_simulation_with_learning_progress(self, free_port):
        """Simulate how the trainer would use curriculum with learning progress."""
        curriculum = StatefulCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start(background=True)
        time.sleep(0.5)

        try:
            # Simulate master rank
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=100
            )

            # Simulate training loop
            for epoch in range(5):
                # Get tasks for environments
                tasks = []
                for _ in range(20):
                    task = client.get_task()
                    tasks.append(task)

                # Simulate completing tasks with scores
                for task in tasks:
                    score = 0.7 + (epoch * 0.05)  # Improving over time
                    # Extract task number from name to complete on server
                    task_num = int(task.name())
                    if curriculum.completed_tasks:
                        actual_task_name = curriculum.completed_tasks[0][0]
                    else:
                        actual_task_name = f"easy_task_{task_num}"
                    curriculum.complete_task(actual_task_name, score)

                # Query stats from client - should be empty
                assert client.stats() == {}

                # Get stats from server-side curriculum
                curriculum_stats = {}
                raw_curriculum_stats = curriculum.stats()
                for key, value in raw_curriculum_stats.items():
                    curriculum_stats[f"curriculum/{key}"] = value

                task_probs = curriculum.get_task_probs()
                for task_id, prob in task_probs.items():
                    curriculum_stats[f"curriculum/task_prob/{task_id}"] = prob

                completion_rates = curriculum.get_completion_rates()
                curriculum_stats.update(completion_rates)

                # Verify stats are being collected
                assert "curriculum/total_tasks" in curriculum_stats
                assert "curriculum/completed_tasks" in curriculum_stats
                assert "curriculum/average_score" in curriculum_stats
                assert "curriculum/task_prob/easy" in curriculum_stats
                assert "curriculum/task_prob/hard" in curriculum_stats

                # Verify progression - task probabilities should shift towards hard
                if epoch > 2:
                    assert task_probs["hard"] > 0.3

        finally:
            client.stop()
            server.stop()

    def test_trainer_with_curriculum_server_client(self, free_port):
        """Test trainer integration with curriculum server and client."""
        # Start a curriculum server
        curriculum = StatefulCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start(background=True)
        time.sleep(0.5)

        try:
            # Create client (as trainer would)
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=10
            )

            # Simulate trainer workflow
            # 1. Get tasks for rollouts
            tasks = []
            for _ in range(5):
                task = client.get_task()
                assert task is not None
                tasks.append(task)

            # 2. Check client stats (should return empty)
            assert client.stats() == {}

            # 3. Complete task (no-op on client, handled by trainer)
            client.complete_task("task_0", 0.8)

            # 4. Verify server curriculum has stats
            server_stats = curriculum.stats()
            assert server_stats["total_tasks"] >= 5

            # 5. Verify tasks have proper env configs
            for task in tasks:
                cfg = task.env_cfg()
                assert cfg is not None
                assert "game" in cfg
                assert cfg.game.num_agents == 2

        finally:
            client.stop()
            server.stop()

    def test_batch_exhaustion_with_trainer_workflow(self, free_port):
        """Test that client handles batch exhaustion like trainer would."""
        # Use a curriculum that generates unique tasks
        curriculum = MockCurriculum()
        server = CurriculumServer(curriculum, port=free_port)
        server.start(background=True)
        time.sleep(0.5)

        try:
            # Small batch size to test prefetching
            client = CurriculumClient(
                server_url=f"http://127.0.0.1:{free_port}",
                batch_size=3,
                prefetch_threshold=0.5,  # Prefetch when queue drops to 1.5 tasks
            )

            # Give time for initial fetch
            time.sleep(0.5)

            # Get many tasks - this should trigger multiple fetches
            all_tasks = []
            for _ in range(30):
                task = client.get_task()
                all_tasks.append(task.name())

            # We should see tasks from multiple batches due to background prefetching
            unique_tasks = set(all_tasks)

            # The server assigns numeric IDs, so we should see many unique ones
            assert len(unique_tasks) >= 10, f"Expected at least 10 unique tasks, got {len(unique_tasks)}"

        finally:
            client.stop()
            server.stop()

    # Helper methods
    def _create_mock_losses(self):
        losses = MagicMock()
        losses.policy_loss = 0.1
        losses.value_loss = 0.2
        losses.entropy = 0.05
        losses.explained_variance = 0.9
        losses.approx_kl_sum = 0.01
        losses.minibatches_processed = 4
        losses.stats = MagicMock(return_value={
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 0.05,
            "explained_variance": 0.9,
            "approx_kl": 0.01 / 4,
            "clipfrac": 0.0,
        })
        return losses

    def _create_mock_experience(self):
        experience = MagicMock()
        experience.num_minibatches = 4
        experience.stats = MagicMock(return_value={
            "buffer_size": 1000,
            "num_episodes": 10
        })
        return experience

    def _create_mock_trainer_config(self):
        trainer_config = MagicMock()
        trainer_config.kickstart.enabled = False
        trainer_config.ppo.l2_reg_loss_coef = 0
        trainer_config.ppo.l2_init_loss_coef = 0
        return trainer_config

    def _create_mock_timer(self):
        timer = MagicMock()
        timer.get_elapsed = MagicMock(return_value=100.0)
        timer.get_last_elapsed = MagicMock(return_value=1.0)
        timer.get_all_elapsed = MagicMock(return_value={
            "_rollout": 50.0,
            "_train": 30.0
        })
        timer.lap_all = MagicMock(return_value={
            "global": 10.0,
            "rollout": 5.0,
            "train": 3.0
        })
        timer.get_lap_steps = MagicMock(return_value=100)
        timer.get_rate = MagicMock(return_value=10.0)
        return timer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])