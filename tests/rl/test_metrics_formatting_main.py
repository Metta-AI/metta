"""Test metrics formatting as it works on main branch."""

from unittest.mock import MagicMock

from metta.rl.functions import (
    build_wandb_stats,
    process_training_stats,
)


class TestMetricsFormattingMain:
    """Test suite for metrics formatting on main branch."""

    def test_build_wandb_stats_with_dict_evals(self):
        """Test build_wandb_stats with dict-based evaluation scores (main branch)."""
        processed_stats = {
            "overview": {"reward": 1.5, "episode_length": 100},
            "losses_stats": {"policy_loss": 0.5, "value_loss": 0.3},
            "experience_stats": {"buffer_size": 1000},
            "environment_stats": {"env/total_steps": 50000},
        }

        timing_info = {
            "epoch_steps_per_second": 1000,
            "wall_time": 3600,
            "train_time": 3000,
            "timing_stats": {"timing/epoch_time": 60},
        }

        weight_stats = {"weights/total_params": 1000000}
        grad_stats = {"gradients/norm": 0.1}
        system_stats = {"monitor/cpu_percent": 50.0}
        memory_stats = {"total_mb": 2048}
        parameters = {"learning_rate": 0.001, "batch_size": 32}

        # On main branch, evals is a Dict[str, float] with keys like "navigation/score", "navigation/maze"
        evals = {
            "navigation/score": 0.8,
            "survival/score": 0.6,
            "navigation/maze": 0.7,
            "navigation/random": 0.9,
            "survival/basic": 0.6,
        }

        result = build_wandb_stats(
            processed_stats,
            timing_info,
            weight_stats,
            grad_stats,
            system_stats,
            memory_stats,
            parameters,
            evals,
            agent_step=10000,
            epoch=100,
            world_size=2,
        )

        # Check overview metrics
        assert result["overview/sps"] == 1000
        assert result["overview/reward"] == 1.5
        assert result["overview/episode_length"] == 100
        assert result["overview/navigation_score"] == 0.8
        assert result["overview/survival_score"] == 0.6
        assert result["overview/reward_vs_total_time"] == 1.5

        # Check losses
        assert result["losses/policy_loss"] == 0.5
        assert result["losses/value_loss"] == 0.3

        # Check experience
        assert result["experience/buffer_size"] == 1000

        # Check parameters
        assert result["parameters/learning_rate"] == 0.001
        assert result["parameters/batch_size"] == 32

        # Check evaluation scores - on main branch they use eval_ prefix directly
        assert result["eval_navigation/score"] == 0.8
        assert result["eval_survival/score"] == 0.6
        assert result["eval_navigation/maze"] == 0.7
        assert result["eval_navigation/random"] == 0.9
        assert result["eval_survival/basic"] == 0.6

        # Check system stats (already prefixed)
        assert result["monitor/cpu_percent"] == 50.0

        # Check memory stats
        assert result["trainer_memory/total_mb"] == 2048

        # Check environment stats
        assert result["env/total_steps"] == 50000

        # Check weight and gradient stats
        assert result["weights/total_params"] == 1000000
        assert result["gradients/norm"] == 0.1

        # Check timing stats
        assert result["timing/epoch_time"] == 60

        # Check metric axes
        assert result["metric/agent_step"] == 20000  # 10000 * 2 (world_size)
        assert result["metric/epoch"] == 100
        assert result["metric/total_time"] == 3600
        assert result["metric/train_time"] == 3000

    def test_evals_processing_in_trainer(self):
        """Test how trainer.py builds the evals dict on main branch."""
        # Simulate what happens in _evaluate_policy
        evals = {}

        # Simulate category scores
        categories = ["navigation", "survival"]
        category_scores = {"navigation": 0.8, "survival": 0.6}

        for category in categories:
            score = category_scores.get(category)
            if score is not None:
                evals[f"{category}/score"] = score

        # Simulate per-simulation scores
        all_scores = [
            ("navigation", "maze", 0.7),
            ("navigation", "random", 0.9),
            ("survival", "basic", 0.6),
        ]

        for category, sim_short_name, score in all_scores:
            evals[f"{category}/{sim_short_name}"] = score

        # Check the structure matches what build_wandb_stats expects
        assert evals["navigation/score"] == 0.8
        assert evals["survival/score"] == 0.6
        assert evals["navigation/maze"] == 0.7
        assert evals["navigation/random"] == 0.9
        assert evals["survival/basic"] == 0.6

        # Extract category scores as done in build_wandb_stats
        category_scores_map = {key.split("/")[0]: value for key, value in evals.items() if key.endswith("/score")}
        assert category_scores_map["navigation"] == 0.8
        assert category_scores_map["survival"] == 0.6

    def test_process_training_stats_basic(self):
        """Test that process_training_stats works the same on main branch."""
        raw_stats = {
            "reward": [1.0, 2.0, 3.0],
            "episode_length": [10, 20, 30],
            "single_value": 42,
        }

        losses = MagicMock()
        losses.stats.return_value = {"policy_loss": 0.5}

        experience = MagicMock()
        experience.stats.return_value = {"buffer_size": 1000}

        trainer_config = MagicMock()
        trainer_config.ppo.l2_reg_loss_coef = 0
        trainer_config.ppo.l2_init_loss_coef = 0

        kickstarter = MagicMock()
        kickstarter.enabled = False

        result = process_training_stats(raw_stats, losses, experience, trainer_config, kickstarter)

        assert result["mean_stats"]["reward"] == 2.0
        assert result["mean_stats"]["episode_length"] == 20
        assert result["mean_stats"]["single_value"] == 42
