from collections import defaultdict
from types import SimpleNamespace

import torch
from tensordict import NonTensorData, TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.rl.loss.loss import Loss, LossConfig
from metta.rl.loss.ppo import PPO, PPOConfig
from metta.rl.loss.sliced_kickstarter import SlicedKickstarter
from metta.rl.loss.sliced_scripted_cloner import SlicedScriptedCloner


class _StubPolicy:
    def __init__(self, experience_spec: Composite):
        self._spec = experience_spec
        self.forward_called = False
        self.burn_in_steps = 0

    def get_agent_experience_spec(self) -> Composite:
        return self._spec

    def reset_memory(self) -> None:
        return None

    def forward(self, td, *, action=None):  # pragma: no cover - should not be hit in empty-mb test
        self.forward_called = True
        raise RuntimeError("policy.forward should not be called for empty minibatches")


class _DummyLoss(Loss):
    def __init__(self):
        base = Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            loss_profile_id=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            is_trainable_agent=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool),
        )
        stub_policy = _StubPolicy(base)
        super().__init__(
            policy=stub_policy,
            trainer_cfg=None,
            env=None,
            device=torch.device("cpu"),
            instance_name="dummy",
            cfg=LossConfig(),
        )  # type: ignore[arg-type]
        self.policy_experience_spec = base
        self._zero_tensor = torch.tensor(0.0)
        self.loss_profiles = {1}
        self.trainable_only = True

    def run_train(self, shared_loss_data, context, mb_idx):
        # Return number of rows seen to validate filtering
        mb = shared_loss_data["sampled_mb"]
        return torch.tensor(float(mb.batch_size.numel())), shared_loss_data, False


def test_loss_profile_and_trainable_filtering():
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
        },
        batch_size=[],
    )
    filtered = loss._filter_minibatch(shared)
    assert filtered["sampled_mb"].batch_size == torch.Size([2])
    assert filtered["sampled_mb"]["actions"].tolist() == [1, 4]
    assert filtered["advantages"].tolist() == [1.0, 4.0]
    assert shared["advantages"].shape[0] == 6  # unchanged input


def test_slot_mask_reduces_2d_layout_to_segments():
    loss = _DummyLoss()
    mb = TensorDict(
        {
            "actions": torch.arange(6).view(3, 2),
            "loss_profile_id": torch.tensor([[1, 1], [0, 2], [1, 0]]),
            "is_trainable_agent": torch.tensor([[True, True], [False, False], [False, False]]),
        },
        batch_size=[3, 2],
    )

    shared = TensorDict(
        {
            "sampled_mb": mb,
            "advantages": torch.arange(6, dtype=torch.float32).view(3, 2),
            "indices": NonTensorData(torch.arange(3)),
        },
        batch_size=[],
    )

    filtered = loss._filter_minibatch(shared)

    assert filtered["sampled_mb"].batch_size == torch.Size([1, 2])
    assert torch.equal(filtered["sampled_mb"]["actions"], torch.tensor([[0, 1]]))
    assert torch.equal(filtered["advantages"], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(filtered["indices"].data, torch.tensor([0]))


class _PPOForTest(PPO):
    def __init__(self):
        spec = Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            loss_profile_id=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            is_trainable_agent=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool),
        )
        policy = _StubPolicy(spec)
        cfg = PPOConfig()
        trainer_cfg = SimpleNamespace()
        super().__init__(
            policy=policy,
            trainer_cfg=trainer_cfg,
            env=None,  # not used in this test
            device=torch.device("cpu"),
            instance_name="ppo",
            loss_config=cfg,
        )
        self.loss_profiles = {1}
        self._stub_minibatch = TensorDict(
            {
                "actions": torch.zeros((2, 1), dtype=torch.int64),
                "loss_profile_id": torch.zeros((2, 1), dtype=torch.int64),
                "is_trainable_agent": torch.ones((2, 1), dtype=torch.bool),
            },
            batch_size=[2, 1],
        )
        self._stub_indices = torch.arange(2)
        self._stub_prio = torch.ones((2, 1))

    @property
    def stub_policy(self) -> _StubPolicy:
        return self.policy  # type: ignore[return-value]

    def _sample_minibatch(self, advantages, prio_alpha, prio_beta, mb_idx):
        return self._stub_minibatch.clone(), self._stub_indices.clone(), self._stub_prio.clone()

    def _process_minibatch_update(self, minibatch, policy_td, indices, prio_weights):  # pragma: no cover
        raise RuntimeError("_process_minibatch_update should not run for empty minibatches")


def test_ppo_skips_empty_filtered_minibatch():
    ppo = _PPOForTest()

    loss_val, shared_loss_data, stop = ppo.run_train(TensorDict({}, batch_size=[]), context=None, mb_idx=1)

    assert loss_val.item() == 0.0
    assert not stop
    assert shared_loss_data["sampled_mb"].batch_size[0] == 0
    assert shared_loss_data["indices"].data.numel() == 0
    assert not ppo.stub_policy.forward_called


class _StubStudentPolicy:
    def __init__(self, num_actions: int = 3):
        self.num_actions = num_actions
        self.forward_called = False

    def reset_memory(self) -> None:
        return None

    def forward(self, td, *, action=None):
        self.forward_called = True
        batch = td.batch_size.numel()
        td["logits"] = torch.zeros(batch, self.num_actions)
        td["values"] = torch.zeros(batch)
        td["full_log_probs"] = torch.zeros(batch, self.num_actions)
        return td


def _stub_replay_from_minibatch(minibatch: TensorDict):
    return SimpleNamespace(
        minibatch_segments=minibatch.batch_size[0],
        segments=minibatch.batch_size[0],
        num_minibatches=1,
        device=minibatch.device,
        buffer=minibatch,
    )


class _TestSlicedKickstarter(SlicedKickstarter):
    """Subclass with __dict__ to simplify testing without running heavy __init__."""

    __slots__ = ("__dict__",)


class _TestSlicedScriptedCloner(SlicedScriptedCloner):
    """Subclass with __dict__ to simplify testing without running heavy __init__."""

    __slots__ = ("__dict__",)


def test_sliced_kickstarter_filters_after_sampling():
    minibatch = TensorDict(
        {
            "actions": torch.tensor([[0], [1]]),
            "teacher_logits": torch.zeros(2, 1, 3),
            "teacher_values": torch.zeros(2, 1),
            "stud_mask": torch.tensor([[True], [False]]),
            "teacher_mask": torch.tensor([[False], [True]]),
            "ppo_mask": torch.tensor([[True], [True]]),
            "loss_profile_id": torch.tensor([[1], [0]]),
            "is_trainable_agent": torch.ones(2, 1, dtype=torch.bool),
        },
        batch_size=[2, 1],
    )

    loss = object.__new__(_TestSlicedKickstarter)
    loss.device = torch.device("cpu")
    loss._zero_tensor = torch.tensor(0.0)
    loss.loss_tracker = defaultdict(list)
    loss.cfg = SimpleNamespace(
        action_loss_coef=1.0,
        value_loss_coef=1.0,
        temperature=1.0,
        teacher_led_proportion=0.0,
        student_led_proportion=0.0,
        student_forward=True,
    )
    loss.policy_experience_spec = Composite(actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64))
    loss.policy = _StubStudentPolicy()
    loss.replay = _stub_replay_from_minibatch(minibatch)
    loss.loss_profiles = {1}

    loss_val, shared_loss_data, stop = loss.run_train(TensorDict({}, batch_size=[]), context=None, mb_idx=0)

    assert not stop
    assert shared_loss_data["sampled_mb"].batch_size[0] == 1  # profile 0 row removed
    assert shared_loss_data["indices"].data.tolist() == [0]
    assert loss.policy.forward_called
    assert loss_val.isfinite()


def test_sliced_scripted_cloner_filters_after_sampling():
    minibatch = TensorDict(
        {
            "actions": torch.tensor([[0], [1]]),
            "teacher_actions": torch.tensor([[1], [2]]),
            "stud_mask": torch.tensor([[True], [False]]),
            "teacher_mask": torch.tensor([[False], [True]]),
            "ppo_mask": torch.tensor([[True], [True]]),
            "loss_profile_id": torch.tensor([[1], [0]]),
            "is_trainable_agent": torch.ones(2, 1, dtype=torch.bool),
        },
        batch_size=[2, 1],
    )

    loss = object.__new__(_TestSlicedScriptedCloner)
    loss.device = torch.device("cpu")
    loss._zero_tensor = torch.tensor(0.0)
    loss.loss_tracker = defaultdict(list)
    loss.cfg = SimpleNamespace(
        action_loss_coef=1.0,
        teacher_led_proportion=0.0,
        student_led_proportion=0.0,
        student_forward=True,
    )
    loss.policy_experience_spec = Composite(actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64))
    loss.policy = _StubStudentPolicy()
    loss.replay = _stub_replay_from_minibatch(minibatch)
    loss.loss_profiles = {1}

    loss_val, shared_loss_data, stop = loss.run_train(TensorDict({}, batch_size=[]), context=None, mb_idx=0)

    assert not stop
    assert shared_loss_data["sampled_mb"].batch_size[0] == 1  # filtered to profile 1 rows
    assert shared_loss_data["indices"].data.tolist() == [0]
    assert loss.policy.forward_called
    assert loss_val.isfinite()
