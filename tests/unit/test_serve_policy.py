from fastapi.testclient import TestClient

from metta.sim.serve_policy import PolicyService, create_app
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.simulator import Action, AgentObservation


class ConstantActionAgentPolicy(AgentPolicy):
    """AgentPolicy that always returns a fixed action_id."""

    def __init__(self, action_id: int):
        # Skip super().__init__ - we don't have a PolicyEnvInterface
        self.action_id = action_id

    def step(self, obs: AgentObservation) -> Action:
        # Return action_id as name (will be parsed back to int by serve_policy)
        return Action(name=str(self.action_id))


class ConstantActionPolicy(MultiAgentPolicy):
    """Policy that always returns a fixed action_id. Useful for testing."""

    def __init__(self, action_id: int):
        # Skip super().__init__ - we don't have a PolicyEnvInterface
        self.action_id = action_id

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return ConstantActionAgentPolicy(self.action_id)


def test_prepare_policy():
    app = create_app(PolicyService(ConstantActionPolicy(42)))
    client = TestClient(app)

    response = client.post(
        "/metta.protobuf.sim.policy_v1.Policy/PreparePolicy",
        json={
            "episode_id": "ep-123",
            "game_rules": {
                "features": [{"id": 1, "name": "health", "normalization": 1.0}],
                "actions": [{"id": 0, "name": "noop"}],
            },
            "agent_ids": [0, 1],
            "observations_format": "TRIPLET_V1",
        },
    )

    assert response.status_code == 200
    assert response.json() == {}


def test_prepare_policy_wrong_path():
    app = create_app(PolicyService(ConstantActionPolicy(42)))
    client = TestClient(app)

    response = client.post("/wrong/path", json={})

    assert response.status_code == 404


def test_prepare_policy_wrong_method():
    app = create_app(PolicyService(ConstantActionPolicy(42)))
    client = TestClient(app)

    response = client.get("/metta.protobuf.sim.policy_v1.Policy/PreparePolicy")

    assert response.status_code == 405


def test_prepare_policy_invalid_json_shape():
    app = create_app(PolicyService(ConstantActionPolicy(42)))
    client = TestClient(app)

    response = client.post(
        "/metta.protobuf.sim.policy_v1.Policy/PreparePolicy",
        json={"episode_id": True},  # should be string, not bool
    )

    assert response.status_code == 400


def test_batch_step():
    app = create_app(PolicyService(ConstantActionPolicy(42)))
    client = TestClient(app)

    # Must call PreparePolicy first to register the episode
    response = client.post(
        "/metta.protobuf.sim.policy_v1.Policy/PreparePolicy",
        json={"episode_id": "ep-123", "agent_ids": [0]},
    )
    assert response.status_code == 200

    response = client.post(
        "/metta.protobuf.sim.policy_v1.Policy/BatchStep",
        json={
            "episode_id": "ep-123",
            "step_id": 1,
            "agent_observations": [
                {"agent_id": 0, "observations": ""},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "agent_actions": [
            {"agent_id": 0, "action_id": [42]},
        ],
    }


def test_batch_step_unknown_episode():
    app = create_app(PolicyService(ConstantActionPolicy(42)))
    client = TestClient(app)

    response = client.post(
        "/metta.protobuf.sim.policy_v1.Policy/BatchStep",
        json={
            "episode_id": "nonexistent",
            "step_id": 1,
            "agent_observations": [],
        },
    )

    assert response.status_code == 404


def test_batch_step_unknown_agent():
    app = create_app(PolicyService(ConstantActionPolicy(42)))
    client = TestClient(app)

    # Register episode with agent 0
    response = client.post(
        "/metta.protobuf.sim.policy_v1.Policy/PreparePolicy",
        json={"episode_id": "ep-123", "agent_ids": [0]},
    )
    assert response.status_code == 200

    # Request step for agent 99, which wasn't registered
    response = client.post(
        "/metta.protobuf.sim.policy_v1.Policy/BatchStep",
        json={
            "episode_id": "ep-123",
            "step_id": 1,
            "agent_observations": [
                {"agent_id": 99, "observations": ""},
            ],
        },
    )

    assert response.status_code == 404
