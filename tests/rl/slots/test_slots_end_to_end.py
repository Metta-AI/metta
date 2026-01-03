from types import SimpleNamespace

import pytest
import torch
from tensordict import NonTensorData, TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.slot_config import PolicySlotConfig
from metta.rl.slot_controller import SlotControllerPolicy

# ---------- Shared stubs ----------


class _StubPolicy:
    def __init__(self, experience_spec: Composite):
        self._spec = experience_spec
        self.forward_called = False
        self.burn_in_steps = 0

    def get_agent_experience_spec(self) -> Composite:
        return self._spec

    def reset_memory(self) -> None:
        return None

    def forward(self, td, *, action=None):
        self.forward_called = True
        td["act_log_prob"] = torch.zeros_like(td["actions"], dtype=torch.float32)
        td["entropy"] = torch.zeros_like(td["actions"], dtype=torch.float32)
        td["values"] = torch.zeros_like(td["actions"], dtype=torch.float32)
        td["full_log_probs"] = torch.zeros(td.shape[0], 3, device=td.device)
        td["logits"] = torch.zeros(td.shape[0], 3, device=td.device)
        return td


class _DummyActions:
    def actions(self):
        return []


def _env_info(num_agents: int) -> SimpleNamespace:
    return SimpleNamespace(
        actions=_DummyActions(),
        observation_space=SimpleNamespace(shape=(1,)),
        num_agents=num_agents,
    )


# ---------- Filtering / metadata ----------


class _DummyLoss(Loss):
    def __init__(self):
        base = Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            loss_profile_id=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            is_trainable_agent=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool),
        )
        policy = _StubPolicy(base)
        super().__init__(
            policy=policy,
            trainer_cfg=None,
            env=None,
            device=torch.device("cpu"),
            instance_name="dummy",
            cfg=LossConfig(),
        )
        self.policy_experience_spec = base
        self._zero_tensor = torch.tensor(0.0)
        self.loss_profiles = {1}
        self.trainable_only = True

    def run_train(self, shared_loss_data, context, mb_idx):
        return torch.tensor(0.0), shared_loss_data, False


def test_loss_filtering_1d_and_nontensor():
    loss = _DummyLoss()
    mb = TensorDict(
        {
            "actions": torch.arange(6),
            "loss_profile_id": torch.tensor([0, 1, 1, 2, 1, 0]),
            "is_trainable_agent": torch.tensor([True, True, False, True, True, False]),
        },
        batch_size=[6],
    )
    shared = TensorDict(
        {
            "sampled_mb": mb,
            "advantages": torch.arange(6, dtype=torch.float32),
            "indices": NonTensorData(torch.arange(6)),
        },
        batch_size=[],
    )
    filtered = loss._filter_minibatch(shared)
    assert filtered["sampled_mb"].batch_size == torch.Size([2])
    assert filtered["sampled_mb"]["actions"].tolist() == [1, 4]
    assert filtered["advantages"].tolist() == [1.0, 4.0]
    assert filtered["indices"].data.tolist() == [1, 4]


def test_loss_filtering_2d_and_cuda_indices():
    if not torch.cuda.is_available():  # pragma: no cover
        pytest.skip("CUDA not available")
    loss = _DummyLoss()
    mb = TensorDict(
        {
            "actions": torch.arange(6, device="cuda").view(3, 2),
            "loss_profile_id": torch.tensor([[1, 1], [0, 2], [1, 0]], device="cuda"),
            "is_trainable_agent": torch.tensor([[True, True], [False, False], [False, False]], device="cuda"),
        },
        batch_size=[3, 2],
    )
    shared = TensorDict(
        {
            "sampled_mb": mb,
            "indices": NonTensorData(torch.arange(3, device="cuda")),
        },
        batch_size=[],
    )
    filtered = loss._filter_minibatch(shared)
    assert filtered["sampled_mb"].batch_size == torch.Size([1, 2])
    assert filtered["indices"].data.is_cuda
    assert filtered["indices"].data.tolist() == [0]


# ---------- Slot controller routing ----------


def test_slot_controller_requires_slot_id_or_map():
    spec = Composite(actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64))
    policy = _StubPolicy(spec)
    env_info = _env_info(1)
    controller = SlotControllerPolicy(
        slot_lookup={"a": 0},
        slots=[],
        slot_policies={0: policy},
        policy_env_info=env_info,
        agent_slot_map=torch.tensor([0]),
    ).to("cpu")
    td = TensorDict(
        {
            "actions": torch.zeros(1, dtype=torch.int64),
            "slot_id": torch.tensor([0]),
            "act_log_prob": torch.zeros(1),
            "entropy": torch.zeros(1),
            "values": torch.zeros(1),
            "full_log_probs": torch.zeros(1, 3),
            "logits": torch.zeros(1, 3),
        },
        batch_size=[1],
    )
    out = controller.forward(td)
    assert "actions" in out


def test_frozen_slot_no_grad():
    td = TensorDict(
        {
            "slot_id": torch.tensor([0, 1]),
            "actions": torch.zeros(2, dtype=torch.int64),
            "act_log_prob": torch.zeros(2),
            "entropy": torch.zeros(2),
            "values": torch.zeros(2),
            "full_log_probs": torch.zeros(2, 3),
            "logits": torch.zeros(2, 3),
        },
        batch_size=[2],
    )
    spec = Composite(actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64))
    grad_policy = _StubPolicy(spec)
    frozen_policy = _StubPolicy(spec)
    frozen_policy.parameters = lambda: [torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)]
    env_info = _env_info(2)
    controller = SlotControllerPolicy(
        slot_lookup={"grad": 0, "frozen": 1},
        slots=[],
        slot_policies={0: grad_policy, 1: frozen_policy},
        policy_env_info=env_info,
        agent_slot_map=torch.tensor([0, 1]),
    ).to("cpu")
    controller.forward(td)
    assert frozen_policy.forward_called


# ---------- Trainer metadata / profiles ----------


def test_slot_metadata_injection_single_agent():
    td = TensorDict({}, batch_size=[1])
    ctx = SimpleNamespace(
        env=SimpleNamespace(policy_env_info=SimpleNamespace(num_agents=1)),
        slot_id_per_agent=torch.tensor([3]),
        loss_profile_id_per_agent=torch.tensor([2]),
        trainable_agent_mask=torch.tensor([True]),
    )
    loop = SimpleNamespace(context=ctx)
    from metta.rl.training.core import CoreTrainingLoop

    CoreTrainingLoop._inject_slot_metadata(loop, td, slice(0, 1))
    assert torch.equal(td["slot_id"], torch.tensor([3]))
    assert torch.equal(td["loss_profile_id"], torch.tensor([2]))
    assert torch.equal(td["is_trainable_agent"], torch.tensor([True]))


def test_action_supervisor_profile_alias():
    from metta.rl.trainer import Trainer

    trainer_cfg = SimpleNamespace()
    trainer_cfg.losses = SimpleNamespace(supervisor=SimpleNamespace(profiles=["teach"]), _configs=lambda: {})
    trainer_cfg.loss_profiles = {"teach": SimpleNamespace(losses=["action_supervisor"])}

    trainer = object.__new__(Trainer)
    trainer._cfg = trainer_cfg

    losses = {"action_supervisor": _DummyLoss()}
    trainer._assign_loss_profiles(losses, {"teach": 7})
    assert losses["action_supervisor"].loss_profiles == {7}


def test_profile_config_drives_loss_filtering():
    from metta.rl.trainer import Trainer

    trainer_cfg = SimpleNamespace()
    trainer_cfg.losses = SimpleNamespace(_configs=lambda: {})
    trainer_cfg.loss_profiles = {
        "teacher_only": SimpleNamespace(losses=["sliced_kickstarter"]),
        "ppo_only": SimpleNamespace(losses=["ppo_actor"]),
    }

    trainer = object.__new__(Trainer)
    trainer._cfg = trainer_cfg

    losses = {
        "sliced_kickstarter": _DummyLoss(),
        "ppo_actor": _DummyLoss(),
    }
    lookup = {"teacher_only": 1, "ppo_only": 2}
    trainer._assign_loss_profiles(losses, lookup)

    assert losses["sliced_kickstarter"].loss_profiles == {1}
    assert losses["ppo_actor"].loss_profiles == {2}


# ---------- Sim runner wiring ----------


def test_sim_runner_uses_device_object(monkeypatch):
    from metta.sim.runner import SimulationRunConfig, run_simulations
    from mettagrid import MettaGridConfig

    class _DummyRegistry:
        def get(self, _slot, _env_info, device):
            assert str(device) == "cpu"
            return _StubPolicy(Composite(actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64)))

    CAPTURED = []

    class _CaptureController:
        def __init__(self, **kwargs):
            CAPTURED.append(kwargs)

        def reset_memory(self):
            return None

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

    slots = [
        PolicySlotConfig(
            id="main",
            class_path="dummy.module:Cls",
            use_trainer_policy=False,
            trainable=True,
            policy_kwargs={},
            policy_uri=None,
        )
    ]
    sim_cfg = SimulationRunConfig(env=MettaGridConfig(), num_episodes=1, policy_slots=slots)

    CAPTURED.clear()
    run_simulations(policy_specs=None, simulations=[sim_cfg], replay_dir=None, seed=0)
    assert CAPTURED
    assert str(CAPTURED[-1]["agent_slot_map"].device) == "cpu"
