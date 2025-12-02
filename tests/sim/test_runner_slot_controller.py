from types import SimpleNamespace

from mettagrid import MettaGridConfig

from metta.rl.slot_config import PolicySlotConfig
from metta.sim.runner import SimulationRunConfig, run_simulations


class _DummyPolicy:
    def __init__(self):
        self.reset_called = False

    def initialize_to_environment(self, *_args, **_kwargs):
        return None

    def reset_memory(self):
        self.reset_called = True

    def forward(self, td, action=None):
        return td

    def get_agent_experience_spec(self):
        return SimpleNamespace()


class _DummyRegistry:
    def get(self, _slot, _env_info, device):
        assert device == "cpu"
        return _DummyPolicy()


CAPTURED_CONTROLLERS = []


class _CaptureController:
    def __init__(self, *, slot_lookup, slots, slot_policies, policy_env_info, device=None, controller_device=None, agent_slot_map=None):
        self.kwargs = dict(
            slot_lookup=slot_lookup,
            slots=slots,
            slot_policies=slot_policies,
            policy_env_info=policy_env_info,
            controller_device=controller_device,
            device=device,
            agent_slot_map=agent_slot_map,
        )
        CAPTURED_CONTROLLERS.append(self)

    def reset_memory(self):
        return None


def test_runner_passes_controller_device(monkeypatch):
    # Stub out heavy dependencies to exercise constructor wiring only
    monkeypatch.setattr("metta.sim.runner.SlotRegistry", lambda: _DummyRegistry())
    monkeypatch.setattr("metta.sim.runner.SlotControllerPolicy", _CaptureController)
    monkeypatch.setattr(
        "metta.sim.runner.PolicyEnvInterface",
        SimpleNamespace(from_mg_cfg=lambda _cfg: SimpleNamespace(num_agents=2)),
    )
    monkeypatch.setattr(
        "metta.sim.runner.multi_episode_rollout",
        lambda **_kwargs: SimpleNamespace(episode_returns=[[1.0, 2.0]], episode_wins=[[1, 0]]),
    )
    monkeypatch.setattr(
        "metta.sim.runner.SimulationRunResult",
        lambda run, results, per_slot_returns, per_slot_winrate: SimpleNamespace(
            run=run,
            results=results,
            per_slot_returns=per_slot_returns,
            per_slot_winrate=per_slot_winrate,
        ),
    )

    slots = [PolicySlotConfig(id="main", class_path="dummy.module:Cls")]
    sim_cfg = SimulationRunConfig(env=MettaGridConfig(), num_episodes=1, policy_slots=slots)

    CAPTURED_CONTROLLERS.clear()
    run_simulations(policy_specs=None, simulations=[sim_cfg], replay_dir=None, seed=0)

    assert CAPTURED_CONTROLLERS, "SlotControllerPolicy should have been constructed"
    kwargs = CAPTURED_CONTROLLERS[-1].kwargs
    assert kwargs["device"] == "cpu"
