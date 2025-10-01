from unittest.mock import MagicMock

from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl import stats as rl_stats
from metta.rl.training.stats_reporter import build_wandb_payload


class TestMetricsFormattingMain:
    def test_build_wandb_stats(self):
        processed_stats = {
            "overview": {"reward": 1.5, "episode_length": 100},
            "losses_stats": {"policy_loss": 0.5, "value_loss": 0.3},
            "experience_stats": {"buffer_size": 1000},
            "environment_stats": {"env/total_steps": 50000},
        }

        timing_info = {
            "epoch_steps_per_second": 1000,
            "steps_per_second": 800,
            "wall_time": 3600,
            "train_time": 3000,
            "timing_stats": {"timing/epoch_time": 60},
        }

        weight_stats = {"weights/total_params": 1000000}
        grad_stats = {"gradients/norm": 0.1}
        system_stats = {"monitor/cpu_percent": 50.0}
        memory_stats = {"total_mb": 2048}
        parameters = {"learning_rate": 0.001, "batch_size": 32}

        evals = EvalRewardSummary(
            category_scores={"navigation": 0.8, "survival": 0.6},
            simulation_scores={
                ("navigation", "maze"): 0.7,
                ("navigation", "random"): 0.9,
                ("survival", "basic"): 0.6,
            },
        )

        hyperparameters = {"learning_rate": 0.001}

        result = build_wandb_payload(
            processed_stats,
            timing_info,
            weight_stats,
            grad_stats,
            system_stats,
            memory_stats,
            parameters,
            hyperparameters,
            evals,
            agent_step=10000,
            epoch=100,
        )

        # Check overview metrics
        assert result["overview/sps"] == 1000
        assert result["overview/reward"] == 1.5
        assert result["overview/episode_length"] == 100
        assert result["overview/navigation_score"] == 0.8
        assert result["overview/survival_score"] == 0.6
        assert result["overview/reward_vs_total_time"] == 1.5
        assert result["overview/steps_per_second"] == 800
        assert result["overview/epoch_steps_per_second"] == 1000

        # Check losses
        assert result["losses/policy_loss"] == 0.5
        assert result["losses/value_loss"] == 0.3

        # Check experience
        assert result["experience/buffer_size"] == 1000

        # Check parameters
        assert result["parameters/learning_rate"] == 0.001
        assert result["parameters/batch_size"] == 32

        # Check hyperparameters
        assert result["hyperparameters/learning_rate"] == 0.001

        # Check evaluation scores
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
        assert result["metric/agent_step"] == 10000
        assert result["metric/epoch"] == 100
        assert result["metric/total_time"] == 3600
        assert result["metric/train_time"] == 3000

    def test_process_training_stats(self):
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

        result = rl_stats.process_training_stats(raw_stats, losses, experience, trainer_config)

        assert result["mean_stats"]["reward"] == 2.0
        assert result["mean_stats"]["episode_length"] == 20
        assert result["mean_stats"]["single_value"] == 42

    def test_process_training_stats_with_dict_values(self):
        """Test that dictionary stats (like per_label_samples) are handled correctly."""
        raw_stats = {
            "reward": [1.0, 2.0, 3.0],
            # Now reports per-epoch deltas instead of cumulative counts
            "env_curriculum/per_label_samples_this_epoch": [
                {"task_A": 100, "task_B": 50},  # Samples in this rollout window
                {"task_A": 120, "task_B": 80},
                {"task_A": 110, "task_B": 70},
            ],
            "env_curriculum/per_label_cumulative_samples": [
                {"task_A": 1000, "task_B": 500},  # Cumulative total for reference
                {"task_A": 1120, "task_B": 580},
                {"task_A": 1230, "task_B": 650},
            ],
            "env_curriculum/per_label_lp_scores": [
                {"task_A": 0.5, "task_B": 0.3},
                {"task_A": 0.6, "task_B": 0.4},
                {"task_A": 0.7, "task_B": 0.5},
            ],
        }

        losses = MagicMock()
        losses.stats.return_value = {}

        experience = MagicMock()
        experience.stats.return_value = {}

        trainer_config = MagicMock()

        result = rl_stats.process_training_stats(raw_stats, losses, experience, trainer_config)

        # Check that per-epoch samples are summed (not averaged) into epoch_samples_per_label/
        assert result["mean_stats"]["epoch_samples_per_label/task_A"] == 330  # Sum: 100 + 120 + 110
        assert result["mean_stats"]["epoch_samples_per_label/task_B"] == 200  # Sum: 50 + 80 + 70

        # Check that averaged per-epoch samples are in mean_samples_per_label/
        assert abs(result["mean_stats"]["mean_samples_per_label/task_A"] - 110.0) < 0.01  # (100 + 120 + 110) / 3
        assert abs(result["mean_stats"]["mean_samples_per_label/task_B"] - 66.666667) < 0.01  # (50 + 80 + 70) / 3

        # Check that cumulative samples are logged separately
        assert result["mean_stats"]["cumulative_samples_per_label/task_A"] == 1230
        assert result["mean_stats"]["cumulative_samples_per_label/task_B"] == 650

        # Check that LP scores are reorganized into epoch_lp_per_label/
        assert abs(result["mean_stats"]["epoch_lp_per_label/task_A"] - 0.7) < 0.01
        assert abs(result["mean_stats"]["epoch_lp_per_label/task_B"] - 0.5) < 0.01

        # Check that averaged LP scores are in mean_lp_per_label/
        assert abs(result["mean_stats"]["mean_lp_per_label/task_A"] - 0.6) < 0.01  # (0.5 + 0.6 + 0.7) / 3
        assert abs(result["mean_stats"]["mean_lp_per_label/task_B"] - 0.4) < 0.01  # (0.3 + 0.4 + 0.5) / 3
