import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.mettagrid.curriculum.colortree_random import ColorTreeRandomFromSetCurriculum
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.mettagrid_env import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)

NUM_AGENTS = 1
OBS_HEIGHT = 3
OBS_WIDTH = 3
NUM_OBS_TOKENS = 32
OBS_TOKEN_SIZE = 3


def create_color_tree_env(
    target_sequence,
    sequence_reward=5.0,
    reward_mode="precise",
    attempts_per_trial=8,
    inventory_item_names=("ore_red", "ore_green", "ore_blue"),
    color_to_item=None,
    max_steps=128,
):
    """Construct a minimal MettaGrid env with ColorTree action enabled."""

    if color_to_item is None:
        # Map first K colors to provided items
        color_to_item = {i: name for i, name in enumerate(inventory_item_names[: max(target_sequence) + 1])}

    game_map = [
        ["wall", "wall", "wall"],
        ["wall", "agent.red", "wall"],
        ["wall", "wall", "wall"],
    ]

    game_config = {
        "max_steps": max_steps,
        "num_agents": NUM_AGENTS,
        "obs_width": OBS_WIDTH,
        "obs_height": OBS_HEIGHT,
        "num_observation_tokens": NUM_OBS_TOKENS,
        "inventory_item_names": list(inventory_item_names),
        "actions": {
            "noop": {"enabled": True},
            "color_tree": {
                "enabled": True,
                "target_sequence": list(target_sequence),
                "sequence_reward": float(sequence_reward),
                "color_to_item": color_to_item,
                "num_trials": 1,
                "trial_sequences": [],
                "attempts_per_trial": int(attempts_per_trial),
                "reward_mode": reward_mode,
            },
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {"wall": {"type_id": 1}},
        "agent": {"rewards": {}},
    }

    env = MettaGrid(from_mettagrid_config(game_config), game_map, 123)

    # Set up buffers
    observations = np.zeros((NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
    terminals = np.zeros(NUM_AGENTS, dtype=dtype_terminals)
    truncations = np.zeros(NUM_AGENTS, dtype=dtype_truncations)
    rewards = np.zeros(NUM_AGENTS, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()
    return env


def perform_color_tree_steps(env: MettaGrid, color_args):
    """Step the env with a sequence of color ids for the 'color_tree' action.
    Returns per-step rewards list and cumulative episode reward from env.
    """
    action_names = env.action_names()
    assert "color_tree" in action_names, f"color_tree not enabled: {action_names}"
    action_idx = action_names.index("color_tree")

    step_rewards = []
    for arg in color_args:
        action = np.zeros((NUM_AGENTS, 2), dtype=dtype_actions)
        action[0, 0] = action_idx
        action[0, 1] = int(arg)
        _, rewards, _, _, _ = env.step(action)
        step_rewards.append(float(rewards[0]))

    return step_rewards, float(env.get_episode_rewards()[0])


class TestColorTreePrecise:
    def test_repeating_correct_sequence_gets_reward_each_window(self):
        # Non-overlapping windows: expect reward on every Nth step
        target = [0, 1, 0]
        env = create_color_tree_env(target_sequence=target, sequence_reward=5.0, reward_mode="precise")

        # Repeat target 3 times => 3 windows
        color_args = target * 3
        step_rewards, episode_total = perform_color_tree_steps(env, color_args)

        # Expect reward only on steps 3,6,9 ... equal to sequence_reward
        expected = [0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0]
        assert step_rewards == pytest.approx(expected, rel=0, abs=1e-6)
        assert episode_total == pytest.approx(15.0, rel=0, abs=1e-6)

    def test_incorrect_sequence_yields_no_reward(self):
        target = [0, 1, 0]
        env = create_color_tree_env(target_sequence=target, sequence_reward=7.0, reward_mode="precise")

        # Repeat incorrect window twice
        wrong = [0, 1, 1] * 2
        step_rewards, episode_total = perform_color_tree_steps(env, wrong)

        assert all(r == 0.0 for r in step_rewards)
        assert episode_total == pytest.approx(0.0, rel=0, abs=1e-6)


class TestColorTreePartial:
    def test_partial_reward_proportional(self):
        target = [0, 1, 2, 3]
        env = create_color_tree_env(target_sequence=target, sequence_reward=4.0, reward_mode="partial")

        # Two windows, each with exactly 2 correct positions out of 4
        # Compare positions: [0,0,2,2] vs target [0,1,2,3] => correct at indices 0 and 2
        actions = [0, 0, 2, 2] * 2
        step_rewards, episode_total = perform_color_tree_steps(env, actions)

        # Reward at the end of each window: 4.0 * (2/4) = 2.0
        expected = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]
        assert step_rewards == pytest.approx(expected, rel=0, abs=1e-6)
        assert episode_total == pytest.approx(4.0, rel=0, abs=1e-6)


class TestColorTreeDense:
    def test_dense_mode_gives_immediate_reward_and_window_completion(self):
        target = [0, 1, 2, 3]
        # Need 4 inventory items to map color 3
        env = create_color_tree_env(
            target_sequence=target,
            sequence_reward=4.0,
            reward_mode="dense",
            inventory_item_names=("ore_red", "ore_green", "ore_blue", "armor"),
        )

        per_pos = 1.0  # sequence_reward / len(target)

        # First 3 steps correct, no window completion yet
        for i in range(3):
            step_rewards, _ = perform_color_tree_steps(env, [target[i]])
            assert step_rewards[0] == pytest.approx(per_pos, rel=0, abs=1e-6)

        # 4th step completes window: per-pos only
        _, rewards, _, _, _ = env.step(
            np.array([[env.action_names().index("color_tree"), target[3]]], dtype=dtype_actions)
        )
        assert rewards[0] == pytest.approx(per_pos, rel=0, abs=1e-6)

        # Episode total equals sequence_reward after full correct window
        assert float(env.get_episode_rewards()[0]) == pytest.approx(4.0, rel=0, abs=1e-6)


def test_colortree_random_sampling_uniform(monkeypatch):
    # Fix RNG for deterministic test of sampling uniformity
    monkeypatch.setattr("os.urandom", lambda n: b"\x00" * 7 + b"\x01")

    # Provide a fake curriculum_from_config_path that returns a single-task curriculum
    def fake_curriculum_from_config_path(path, env_overrides=None):
        base_cfg = OmegaConf.create(
            {
                "game": {
                    "num_agents": 1,
                    "actions": {
                        "color_tree": {
                            "target_sequence": [0, 1],
                            "sequence_reward": 1.0,
                            "color_to_item": {0: "ore_red", 1: "ore_green"},
                            "num_trials": 1,
                            "trial_sequences": [],
                            "attempts_per_trial": 8,
                            "reward_mode": "precise",
                        }
                    },
                }
            }
        )
        return SingleTaskCurriculum(path, base_cfg)

    monkeypatch.setattr(
        "metta.mettagrid.curriculum.random.curriculum_from_config_path",
        fake_curriculum_from_config_path,
    )

    # Create curriculum with 2 colors and length 2 => 4 sequences total
    curr = ColorTreeRandomFromSetCurriculum(tasks={"/dummy": 1.0}, num_colors=2, sequence_length=2)

    counts = {tuple([0, 0]): 0, tuple([0, 1]): 0, tuple([1, 0]): 0, tuple([1, 1]): 0}

    samples = 2000
    for _ in range(samples):
        _ = curr.get_task()
        stats = curr.get_curriculum_stats()
        seq_str = stats["selected_sequence"]
        seq = tuple(int(x) for x in seq_str.split(","))
        counts[seq] = counts.get(seq, 0) + 1

    # All sequences should appear
    assert all(c > 0 for c in counts.values())

    mean = samples / 4
    # Tolerate moderate variance; with fixed seed this should pass comfortably
    for seq, c in counts.items():
        assert 0.8 * mean <= c <= 1.2 * mean, f"Sequence {seq} count {c} out of bounds"


def test_curriculum_aligns_max_steps_to_sequence_windows(monkeypatch):
    # fake task config to attach game config
    def fake_curriculum_from_config_path(path, env_overrides=None):
        base_cfg = OmegaConf.create(
            {
                "game": {
                    "num_agents": 1,
                    "max_steps": 13,  # Intentionally too small and not multiple of sequence_length
                    "actions": {
                        "color_tree": {
                            "target_sequence": [0, 1, 2, 3],
                            "sequence_reward": 1.0,
                            "color_to_item": {0: "ore_red", 1: "ore_green", 2: "ore_blue", 3: "armor"},
                            "num_trials": 1,
                            "trial_sequences": [],
                            "attempts_per_trial": 5,  # requires at least 5 * 4 = 20 steps
                            "reward_mode": "precise",
                        }
                    },
                }
            }
        )
        return SingleTaskCurriculum(path, base_cfg)

    monkeypatch.setattr(
        "metta.mettagrid.curriculum.random.curriculum_from_config_path",
        fake_curriculum_from_config_path,
    )

    # sequence_length is 4; attempts_per_trial is 5 => require at least 20 and align to multiple of 4
    curr = ColorTreeRandomFromSetCurriculum(tasks={"/dummy": 1.0}, num_colors=4, sequence_length=4)
    task = curr.get_task()
    cfg = task.env_cfg()
    # attempts are now computed from max_steps; since starting max_steps was 13 and seq_len=4,
    # computed attempts = floor(13/4) = 3, and we do not increase max_steps here.
    assert cfg.game.actions.color_tree.attempts_per_trial == 3
