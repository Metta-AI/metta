import torch
from tensordict import NonTensorData, TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.rl.loss.loss import Loss, LossConfig


class _StubPolicy:
    def __init__(self, experience_spec: Composite):
        self._spec = experience_spec

    def get_agent_experience_spec(self) -> Composite:
        return self._spec


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
