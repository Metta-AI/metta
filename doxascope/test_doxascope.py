import json

import numpy as np
import pytest
import torch

from metta.agent.policy_state import PolicyState

from .doxascope_data import DoxascopeLogger, Movement, preprocess_doxascope_data
from .doxascope_network import DoxascopeNet
from .doxascope_train import train_doxascope


@pytest.fixture
def doxascope_dirs(tmp_path):
    """Creates a temporary directory structure for a full pipeline test."""
    policy_name = "test_policy"
    raw_data_dir = tmp_path / "raw_data" / policy_name
    results_dir = tmp_path / "results" / policy_name

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create multiple dummy raw data files to ensure the train/val/test split works
    for i in range(10):
        raw_data = []
        num_timesteps = 50
        agent_trajectory = []
        pos = [0, 0]
        for t in range(num_timesteps):
            # Create some movement
            if t % 4 == 0:
                pos[0] += 1
            elif t % 4 == 1:
                pos[1] += 1
            elif t % 4 == 2:
                pos[0] -= 1
            else:
                pos[1] -= 1

            agent_trajectory.append({"agent_id": 0, "memory_vector": [0.1] * 512, "position": list(pos)})

        for t in range(num_timesteps):
            raw_data.append({"timestep": t, "agents": [agent_trajectory[t]]})

        data_file = raw_data_dir / f"doxascope_data_test_{i}.json"
        with open(data_file, "w") as f:
            json.dump(raw_data, f)

    return {
        "policy_name": policy_name,
        "raw_data_dir": raw_data_dir,
        "results_dir": results_dir,
        "tmp_path": tmp_path,
    }


def test_doxascope_training_pipeline(doxascope_dirs):
    """
    An integration test that runs the core training pipeline.
    """
    results = train_doxascope(
        raw_data_dir=doxascope_dirs["raw_data_dir"],
        output_dir=doxascope_dirs["results_dir"],
        num_epochs=1,
        num_future_timesteps=2,
        num_past_timesteps=1,
        test_split=0.5,  # Ensure we have test data
        val_split=0.2,
    )

    assert results is not None, "Training pipeline failed to run."

    results_dir = doxascope_dirs["results_dir"]
    assert (results_dir / "best_model.pth").exists()
    assert (results_dir / "training_curves.png").exists()
    assert (results_dir / "multistep_accuracy.png").exists()
    assert (results_dir / "confusion_matrix_t-1.png").exists()
    assert (results_dir / "confusion_matrix_t+2.png").exists()

    preprocessed_dir = results_dir / "preprocessed_data"
    assert (preprocessed_dir / "train_data.npz").exists()
    assert (preprocessed_dir / "val_data.npz").exists()
    assert (preprocessed_dir / "test_data.npz").exists()


def test_preprocess_doxascope_data(doxascope_dirs):
    """
    Tests the data preprocessing logic to ensure correct parsing of trajectories.
    """
    raw_data_dir = doxascope_dirs["raw_data_dir"]
    preprocessed_dir = doxascope_dirs["results_dir"] / "preprocessed_data"
    json_files = list(raw_data_dir.glob("*.json"))

    num_past = 1
    num_future = 2

    X, y = preprocess_doxascope_data(
        json_files,
        preprocessed_dir,
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
    )

    assert X is not None and y is not None

    # Each of the 10 files has 50 timesteps.
    # The loop runs from i=2 to 50-2-1=47.
    # range(2, 48) -> 46 samples per file. 10 files -> 460 samples.
    expected_samples = (50 - num_past - num_future - 1) * 10
    assert X.shape == (expected_samples, 512)
    assert y.shape == (expected_samples, num_past + num_future)

    # Check the content of a specific y sample based on the fixture's movement pattern.
    # `move(t)` is the delta from pos(t-1) to pos(t).
    # t=0: pos=(0,0) -> This is the initial state before any move.
    # t=1: pos=(1,0), move(1) = pos(1)-pos(0)=(1,0) -> down(2)
    # t=2: pos=(1,1), move(2) = pos(2)-pos(1)=(0,1) -> right(4)
    # t=3: pos=(0,1), move(3) = pos(3)-pos(2)=(-1,0) -> up(1)
    # t=4: pos=(0,0), move(4) = pos(4)-pos(3)=(0,-1) -> left(3)
    # This pattern seems off, let's re-verify the fixture.
    # In fixture: initial pos is (0,0), then pos[0]+=1 -> (1,0) at t=0. Let's trace carefully.
    # t=0: pos=(1,0)
    # t=1: pos=(1,1) -> move(1) = (0,1) = RIGHT
    # t=2: pos=(0,1) -> move(2) = (-1,0) = UP
    # t=3: pos=(0,0) -> move(3) = (0,-1) = LEFT
    # t=4: pos=(1,0) -> move(4) = (1,0) = DOWN
    move_pattern = [Movement.DOWN, Movement.RIGHT, Movement.UP, Movement.LEFT]  # Moves for t=0,1,2,3

    # First sample is from i=num_past+1=2.
    # y should be [move(i-1), move(i+1), move(i+2)] -> [move(1), move(3), move(4)]
    # move(4) is DOWN.
    expected_y_for_i_1 = [move_pattern[1], move_pattern[3], Movement.DOWN]
    np.testing.assert_array_equal(y[0], expected_y_for_i_1)

    # Second sample is from i=3.
    # y should be [move(i-1), move(i+1), move(i+2)] -> [move(2), move(4), move(5)]
    # move(4) is DOWN, move(5) is RIGHT
    expected_y_for_i_2 = [move_pattern[2], Movement.DOWN, Movement.RIGHT]
    np.testing.assert_array_equal(y[1], expected_y_for_i_2)


def test_doxascope_net_forward_pass():
    """
    Tests the forward pass of the DoxascopeNet to ensure correct output shapes.
    """
    batch_size = 4
    input_dim = 512
    num_past = 1
    num_future = 4
    num_classes = 5

    model = DoxascopeNet(
        input_dim=input_dim,
        num_past_timesteps=num_past,
        num_future_timesteps=num_future,
        num_classes=num_classes,
    )
    model.eval()

    dummy_input = torch.randn(batch_size, input_dim)
    outputs = model(dummy_input)

    assert isinstance(outputs, list)
    assert len(outputs) == num_past + num_future

    for out in outputs:
        assert out.shape == (batch_size, num_classes)


def test_doxascope_logger(doxascope_dirs):
    """
    Tests the DoxascopeLogger to ensure it correctly logs and saves data.
    """
    tmp_path = doxascope_dirs["tmp_path"]
    policy_name = "test_logger_policy"
    sim_id = "test_sim_123"

    # 1. Initialize logger
    logger = DoxascopeLogger(
        doxascope_config={"enabled": True, "output_dir": tmp_path / "raw_data"},
        simulation_id=sim_id,
        policy_name=policy_name,
    )
    assert logger.enabled

    # 2. Log a timestep
    policy_state = PolicyState(
        lstm_h=torch.ones(1, 1, 256),
        lstm_c=torch.zeros(1, 1, 256),
        batch_size=[1],
    )
    policy_idxs = torch.tensor([0])
    env_grid_objects = {
        "obj1": {"type": 0, "agent_id": 0, "r": 10, "c": 20},
    }
    logger.log_timestep(policy_state, policy_idxs, env_grid_objects)

    # 3. Save data
    logger.save()

    # 4. Verify output file and content
    expected_file = tmp_path / "raw_data" / policy_name / f"doxascope_data_{sim_id}.json"
    assert expected_file.exists()

    with open(expected_file, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 1
    timestep_data = data[0]
    assert timestep_data["timestep"] == 0
    assert len(timestep_data["agents"]) == 1

    agent_data = timestep_data["agents"][0]
    assert agent_data["agent_id"] == 0
    assert agent_data["position"] == [10, 20]
    assert len(agent_data["memory_vector"]) == 512
    assert agent_data["memory_vector"][:256] == [1.0] * 256
    assert agent_data["memory_vector"][256:] == [0.0] * 256
