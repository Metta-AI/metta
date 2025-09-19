"""Tests for the trainer checkpoint components in the hook-based training stack."""

from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.policy import Policy
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.checkpointer import Checkpointer, CheckpointerConfig
from metta.rl.training.context import TrainerContext, TrainerState
from metta.rl.training.context_checkpointer import ContextCheckpointer, ContextCheckpointerConfig
from metta.rl.training.distributed_helper import DistributedHelper
from mettagrid.profiling.stopwatch import Stopwatch


class DummyPolicy(Policy):
    """Small concrete Policy used for checkpointing tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        td = td.clone(False)
        td["values"] = torch.zeros(td.batch_size.numel(), dtype=torch.float32)
        return td

    def get_agent_experience_spec(self) -> Composite:  # noqa: D401
        return Composite(values=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32))

    def initialize_to_environment(self, env_metadata, device: torch.device) -> None:  # noqa: D401
        return None

    @property
    def device(self) -> torch.device:  # noqa: D401
        return torch.device("cpu")

    @property
    def total_params(self) -> int:  # noqa: D401
        return sum(param.numel() for param in self.parameters())

    def reset_memory(self) -> None:  # noqa: D401
        return None

    def clip_weights(self) -> None:
        return None


def build_context(tmp_path, policy: Policy, optimizer: torch.optim.Optimizer) -> TrainerContext:
    stopwatch = Stopwatch()
    stopwatch.start()
    distributed = DistributedHelper(torch.device("cpu"))

    experience = SimpleNamespace(accumulate_minibatches=1)
    context = TrainerContext(
        state=TrainerState(run_dir=str(tmp_path), run_name="test"),
        policy=policy,
        env=SimpleNamespace(),
        experience=experience,
        optimizer=optimizer,
        config=SimpleNamespace(),
        device=torch.device("cpu"),
        stopwatch=stopwatch,
        distributed=distributed,
    )
    context.get_train_epoch_fn = lambda: (lambda: None)
    context.set_train_epoch_fn = lambda fn: None
    return context


def test_trainer_checkpointer_restore(tmp_path):
    policy = DummyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    context = build_context(tmp_path, policy, optimizer)

    manager = CheckpointManager(run="trainer", run_dir=str(tmp_path))
    manager.save_trainer_state(optimizer, epoch=3, agent_step=128, stopwatch_state={"foo": 1})

    component = ContextCheckpointer(
        config=ContextCheckpointerConfig(epoch_interval=1),
        checkpoint_manager=manager,
        distributed_helper=context.distributed,
    )
    component.register(context)
    component.restore(context)

    assert context.epoch == 3
    assert context.agent_step == 128


def test_trainer_checkpointer_saves_state(tmp_path):
    policy = DummyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    context = build_context(tmp_path, policy, optimizer)

    manager = CheckpointManager(run="trainer", run_dir=str(tmp_path))
    component = ContextCheckpointer(
        config=ContextCheckpointerConfig(epoch_interval=1),
        checkpoint_manager=manager,
        distributed_helper=context.distributed,
    )
    component.register(context)

    context.epoch = 4
    context.agent_step = 256

    component.on_epoch_end(4)
    state = manager.load_trainer_state()
    assert state["epoch"] == 4
    assert state["agent_step"] == 256


def test_checkpointer_updates_latest_uri(tmp_path):
    policy = DummyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    context = build_context(tmp_path, policy, optimizer)
    context.latest_eval_scores = EvalRewardSummary()

    manager = CheckpointManager(run="policy", run_dir=str(tmp_path))
    component = Checkpointer(
        config=CheckpointerConfig(epoch_interval=1),
        checkpoint_manager=manager,
        distributed_helper=context.distributed,
    )
    component.register(context)

    component.on_epoch_end(2)

    latest_uri = context.latest_policy_uri()
    assert latest_uri is not None
    saved_path = Path(latest_uri[7:])
    assert saved_path.exists()
