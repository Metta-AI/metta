import json
import subprocess
import sys

import pytest

from doxascope.doxascope_data import (
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
        "doxascope.doxascope_train",
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

    # --- 3. Run Analysis ---
    analyze_cmd = [
        "doxascope.doxascope_analysis",
        "analyze",
        policy_name,
        "test_run_1",
        "--data-dir",
        str(results_dir),
    ]
    run_command(analyze_cmd)

    # --- 4. Verify Analysis Outputs ---
    analysis_dir = run_dir / "analysis"
    assert analysis_dir.is_dir()
    assert (analysis_dir / "multistep_accuracy.png").exists()
    assert (analysis_dir / "training_history.png").exists()

    # --- 5. Run Comparison ---
    # First, run training again to create a second run to compare with
    run_command(train_cmd[:-2] + ["test_run_2"])

    compare_cmd = [
        "doxascope.doxascope_analysis",
        "compare",
        policy_name,
        "--data-dir",
        str(results_dir),
    ]
    run_command(compare_cmd)

    # --- 6. Verify Comparison Outputs ---
    policy_results_dir = results_dir / policy_name
    assert (policy_results_dir / f"comparison_{policy_name}.png").exists()

    # --- 7. Test analyzing the latest run (no run_name specified) ---
    latest_run_analyze_cmd = [
        "doxascope.doxascope_analysis",
        "analyze",
        policy_name,
        "--data-dir",
        str(results_dir),
    ]
    run_command(latest_run_analyze_cmd)

    # --- 8. Verify Latest Run Analysis Outputs ---
    # Should have analyzed test_run_2 as it is the latest
    latest_run_dir = results_dir / policy_name / "test_run_2"
    latest_analysis_dir = latest_run_dir / "analysis"
    assert latest_analysis_dir.is_dir()
    assert (latest_analysis_dir / "multistep_accuracy.png").exists()
    assert (latest_analysis_dir / "training_history.png").exists()
