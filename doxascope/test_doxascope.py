import json
import subprocess
import sys

import pytest

from doxascope.doxascope_data import (
    DoxascopeLogger,
    class_id_to_pos,
    get_num_classes_for_manhattan_distance,
    get_positions_for_manhattan_distance,
    pos_to_class_id,
)


def test_get_num_classes():
    assert get_num_classes_for_manhattan_distance(0) == 1
    assert get_num_classes_for_manhattan_distance(1) == 5
    assert get_num_classes_for_manhattan_distance(2) == 13
    assert get_num_classes_for_manhattan_distance(3) == 25


def test_get_positions():
    # d=0
    assert get_positions_for_manhattan_distance(0) == [(0, 0)]
    # d=1
    assert get_positions_for_manhattan_distance(1) == [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
    # d=2
    positions_d2 = [
        (-2, 0),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (1, -1),
        (1, 0),
        (1, 1),
        (2, 0),
    ]
    assert get_positions_for_manhattan_distance(2) == positions_d2


@pytest.mark.parametrize(
    "d, pos, expected_id",
    [
        (1, (0, 0), 2),
        (1, (1, 0), 4),
        (1, (-1, 0), 0),
        (2, (0, 0), 6),
        (2, (2, 0), 12),
        (2, (-2, 0), 0),
        (2, (1, 1), 11),
    ],
)
def test_pos_to_class_id(d, pos, expected_id):
    assert pos_to_class_id(pos[0], pos[1], d) == expected_id


@pytest.mark.parametrize(
    "d, class_id, expected_pos",
    [
        (1, 2, (0, 0)),
        (1, 4, (1, 0)),
        (1, 0, (-1, 0)),
        (2, 6, (0, 0)),
        (2, 12, (2, 0)),
        (2, 0, (-2, 0)),
        (2, 11, (1, 1)),
    ],
)
def test_class_id_to_pos(d, class_id, expected_pos):
    assert class_id_to_pos(class_id, d) == expected_pos


def test_round_trip_conversion():
    for d in range(5):
        positions = get_positions_for_manhattan_distance(d)
        for i, pos in enumerate(positions):
            class_id = pos_to_class_id(pos[0], pos[1], d)
            assert class_id == i
            retrieved_pos = class_id_to_pos(class_id, d)
            assert retrieved_pos == pos


def test_invalid_inputs():
    with pytest.raises(ValueError):
        pos_to_class_id(2, 0, 1)  # Outside distance

    with pytest.raises(ValueError):
        class_id_to_pos(99, 1)  # Out of bounds


@pytest.fixture
def doxascope_env(tmp_path):
    """Creates a temporary directory structure that mimics the real one."""
    policy_name = "test_policy"
    base_dir = tmp_path / "train_dir" / "doxascope"
    raw_data_dir = base_dir / "raw_data" / policy_name
    results_dir = base_dir / "results"

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create a few dummy raw data files with a predictable "box" movement pattern
    for i in range(5):
        raw_data = []
        pos = [0, 0]
        agent_trajectory = []
        for t in range(50):
            if t < 10:
                pos[0] += 1  # Move right
            elif t < 20:
                pos[1] += 1  # Move up
            elif t < 30:
                pos[0] -= 1  # Move left
            else:
                pos[1] -= 1  # Move down
            agent_trajectory.append({"agent_id": 0, "memory_vector": [0.1] * 512, "position": list(pos)})

        for t in range(50):
            raw_data.append({"timestep": t, "agents": [agent_trajectory[t]]})

        data_file = raw_data_dir / f"sim_data_{i}.json"
        with open(data_file, "w") as f:
            json.dump(raw_data, f)

    return {
        "policy_name": policy_name,
        "raw_data_dir": base_dir / "raw_data",
        "results_dir": results_dir,
        "base_dir": base_dir,
    }


def run_command(cmd: list):
    """Helper to run a script as a subprocess and handle errors."""
    process = subprocess.run(
        [sys.executable, "-m"] + cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        print("--- STDOUT ---")
        print(process.stdout)
        print("--- STDERR ---")
        print(process.stderr)
        pytest.fail(f"Command '{' '.join(cmd)}' failed with exit code {process.returncode}", pytrace=False)
    return process


def test_doxascope_end_to_end(doxascope_env):
    """
    A full end-to-end integration test for the doxascope module.
    1. Runs the training script to generate a versioned run directory.
    2. Verifies that all expected output files are created.
    3. Runs the analysis script's `analyze` and `compare` commands.
    4. Verifies that the analysis plots are generated.
    """
    policy_name = doxascope_env["policy_name"]
    raw_data_dir = doxascope_env["raw_data_dir"]
    results_dir = doxascope_env["results_dir"]

    # --- 1. Run Training ---
    train_cmd = [
        "doxascope.cli",
        "train",
        policy_name,
        "--raw-data-dir",
        str(raw_data_dir),
        "--output-dir",
        str(results_dir),
        "--num-epochs",
        "2",
        "--num-future-timesteps",
        "1",
        "--run-name",
        "test_run_1",
        "--no-train-random-baseline",  # Disable baseline for speed in this test
    ]
    run_command(train_cmd)

    # --- 2. Verify Training Outputs ---
    run_dir = results_dir / policy_name / "test_run_1"
    assert run_dir.is_dir()
    assert (run_dir / "best_model.pth").exists()
    assert (run_dir / "test_results.json").exists()
    assert (run_dir / "training_history.csv").exists()
    assert (run_dir / "preprocessed_data" / "train.npz").exists()

    # --- 3. Verify Analysis Outputs (generated automatically during training) ---
    analysis_dir = run_dir / "analysis"
    assert analysis_dir.is_dir()
    assert (analysis_dir / "multistep_accuracy_comparison.png").exists()
    assert (analysis_dir / "training_history.png").exists()


def test_logger_agent_alignment():
    """Verify DoxascopeLogger logs per-agent memory aligned with the correct agent positions."""
    import numpy as np
    import torch

    # Setup logger
    logger = DoxascopeLogger(enabled=True, simulation_id="simtest")
    logger.configure(policy_uri="test://policy_a", object_type_names=["agent"])  # agent type id = 0

    # Single env with 2 agents
    num_agents = 2

    # Build grid_objects for env with agent_ids 0, 1
    env_grid_objects = {
        1: {"type": 0, "agent_id": 0, "r": 0, "c": 0},
        2: {"type": 0, "agent_id": 1, "r": 0, "c": 1},
    }

    # Fake policy with LSTM buffers in components['lstm_reset'] matching [L, B, H]
    L, H = 1, 3
    B = num_agents
    lstm_h = torch.zeros((L, B, H), dtype=torch.float32)
    lstm_c = torch.zeros((L, B, H), dtype=torch.float32)
    # Give each agent a distinct constant pattern so we can validate alignment
    for i in range(B):
        lstm_h[:, i, :] = (i + 1) * 1.0
        lstm_c[:, i, :] = (i + 1) * 10.0

    class FakeLSTM:
        def __init__(self, h, c):
            self.lstm_h = h
            self.lstm_c = c

    fake_lstm = FakeLSTM(lstm_h, lstm_c)

    class FakePolicy:
        def __init__(self, components):
            self.components = components

    policy = FakePolicy({"lstm_reset": fake_lstm})

    # Policy indices [0, 1]
    policy_idxs = torch.tensor([0, 1], dtype=torch.long)

    # Log one timestep
    logger.log_timestep(policy, policy_idxs, env_grid_objects)

    # Validate
    assert logger.data, "Logger produced no data entries"
    agents = logger.data[-1]["agents"]
    assert len(agents) == num_agents

    # Expected positions: agent 0 -> (0,0), agent 1 -> (0,1)
    expected_positions = [(0, 0), (0, 1)]
    for i, agent_entry in enumerate(agents):
        assert tuple(agent_entry["position"]) == expected_positions[i]
        # Memory vector expected: concat(h, c) for agent i, flattened
        expected_vec = np.concatenate(
            [(i + 1) * np.ones(H, dtype=np.float32), (i + 1) * 10.0 * np.ones(H, dtype=np.float32)]
        )
        mv = np.array(agent_entry["memory_vector"], dtype=np.float32)
        # The flattened shape is [L*2H] == [2H] since L=1
        assert mv.shape[0] == 2 * H
        assert np.allclose(mv, expected_vec)
