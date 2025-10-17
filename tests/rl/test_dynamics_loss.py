import torch
from tensordict import TensorDict
from torchrl.data import Composite

from metta.rl.loss.dynamics import Dynamics, DynamicsConfig


class _DummyPolicy:
    def get_agent_experience_spec(self) -> Composite:
        return Composite()


def _make_shared_loss_data(*, segments: int, horizon: int, num_actions: int):
    returns = torch.rand(segments, horizon)
    rewards = torch.rand(segments, horizon)
    actions = torch.randint(0, num_actions, (segments, horizon))

    # Align predictions to target future values exactly so the dynamics loss is zero
    returns_pred = returns.clone().unsqueeze(-1)
    reward_pred = rewards.clone().unsqueeze(-1)

    logits = torch.full((segments, horizon, num_actions), -9.0)
    for b in range(segments):
        for t in range(horizon - 1):
            logits[b, t, actions[b, t + 1]] = 5.0
        logits[b, -1] = 0.0  # last step unused

    policy_td = TensorDict(
        {
            "returns_pred": returns_pred,
            "reward_pred": reward_pred,
            "action_pred_logits": logits,
        },
        batch_size=[segments, horizon],
    )

    sampled_mb = TensorDict(
        {
            "returns": returns,
            "rewards": rewards,
            "actions": actions,
            "dones": torch.zeros(segments, horizon, 1, dtype=torch.bool),
        },
        batch_size=[segments, horizon],
    )

    return TensorDict({"policy_td": policy_td, "sampled_mb": sampled_mb}, batch_size=[])


def test_dynamics_loss_includes_action_term():
    cfg = DynamicsConfig(
        returns_step_look_ahead=1,
        returns_pred_coef=1.0,
        reward_pred_coef=1.0,
        action_pred_coef=1.0,
    )

    loss = Dynamics(
        policy=_DummyPolicy(),
        trainer_cfg=object(),
        env=object(),
        device=torch.device("cpu"),
        instance_name="dynamics",
        loss_cfg=cfg,
    )

    shared_loss_data = _make_shared_loss_data(segments=2, horizon=4, num_actions=3)

    total_loss, _, _ = loss.run_train(shared_loss_data, context=type("ctx", (), {"epoch": 0})(), mb_idx=0)

    assert total_loss.item() >= 0.0
    assert loss.loss_tracker["dynamics_action_loss"], "Action loss should be tracked"
