import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.rl.loss.loss import Loss


class _DummyLoss(Loss):
    def __init__(self):
        base = Composite(
            actions=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            loss_profile_id=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.int64),
            is_trainable_agent=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.bool),
        )
        super().__init__(
            policy=None,
            trainer_cfg=None,
            env=None,
            device=torch.device("cpu"),
            instance_name="dummy",
            loss_config=None,
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
    out_val, _, _ = loss.train(shared, None, 0)
    # Expected rows: profile==1 AND trainable -> indices 1 and 4 -> 2 rows
    assert out_val.item() == 2.0
    assert shared["advantages"].shape[0] == 6  # unchanged input


def test_npc_rows_are_filtered_when_not_trainable():
    loss = _DummyLoss()
    mb = TensorDict(
        {
            "actions": torch.arange(4),
            "loss_profile_id": torch.tensor([1, 1, 1, 1]),
            "is_trainable_agent": torch.tensor([True, False, False, True]),
        },
        batch_size=[4],
    )
    shared = TensorDict({"sampled_mb": mb}, batch_size=[])
    out_val, _, _ = loss.train(shared, None, 0)
    # Only two trainable rows should remain
    assert out_val.item() == 2.0
