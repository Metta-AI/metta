from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from metta.agent.components.actor import ActionProbsConfig
from metta.agent.policies.fast import FastConfig
from metta.agent.policy import Policy, PolicyArchitecture
from metta.cogworks.curriculum import env_curriculum
from metta.rl.checkpoint_bundle import write_checkpoint_dir
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import CheckpointerConfig, EvaluatorConfig, TrainingEnvironmentConfig
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_arena
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class DummyPolicyArchitecture(PolicyArchitecture):
    class_path: str = "tests.helpers.fast_train_tool.DummyPolicy"
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, policy_env_info):
        return DummyPolicy(0)


class DummyPolicy(Policy, nn.Module):
    def __init__(self, epoch: int) -> None:
        policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
        super().__init__(policy_env_info)
        self.register_buffer("epoch_tensor", torch.tensor(epoch, dtype=torch.float32))

    def forward(self, td) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def reset_memory(self) -> None:
        pass


class FastCheckpointTrainTool(TrainTool):
    def invoke(self, args: dict[str, str]) -> int | None:
        if "run" in args:
            assert self.run is None, "run cannot be set twice"
            self.run = args["run"]

        run_name = self.run or "default"

        checkpoint_manager = CheckpointManager(run=run_name, system_cfg=self.system)

        trainer_state_path = checkpoint_manager.checkpoint_dir / "trainer_state.pt"
        if trainer_state_path.exists():
            previous_state = torch.load(trainer_state_path, weights_only=False)
            previous_agent_step = int(previous_state.get("agent_step", 0))
            previous_epoch = int(previous_state.get("epoch", 0))
        else:
            previous_agent_step = 0
            previous_epoch = 0

        agent_step = max(previous_agent_step, int(self.trainer.total_timesteps))
        epoch = previous_epoch + 1

        torch.save(
            {
                "agent_step": agent_step,
                "epoch": epoch,
                "optimizer_state": {},
            },
            trainer_state_path,
        )

        policy = DummyPolicy(epoch)
        architecture = DummyPolicyArchitecture()
        write_checkpoint_dir(
            base_dir=checkpoint_manager.checkpoint_dir,
            run_name=run_name,
            epoch=epoch,
            policy_class_path=architecture.class_path,
            architecture_spec=architecture.to_spec(),
            state_dict=policy.state_dict(),
        )

        return 0


def run_fast_train_tool(
    *,
    run_name: str,
    system_cfg: SystemConfig,
    trainer_cfg: TrainerConfig,
    training_env_cfg: TrainingEnvironmentConfig,
    policy_cfg: FastConfig,
    checkpointer: CheckpointerConfig | None = None,
    evaluator: EvaluatorConfig | None = None,
) -> CheckpointManager:
    """Run the fast checkpoint tool once and return the manager for the run."""
    tool = FastCheckpointTrainTool(
        run=run_name,
        system=system_cfg.model_copy(deep=True),
        trainer=trainer_cfg.model_copy(deep=True),
        training_env=training_env_cfg.model_copy(deep=True),
        policy_architecture=policy_cfg.model_copy(deep=True),
        stats_server_uri=None,
        checkpointer=checkpointer or CheckpointerConfig(epoch_interval=1),
        evaluator=evaluator or EvaluatorConfig(epoch_interval=0, evaluate_local=False, evaluate_remote=False),
    )
    tool.invoke({})
    return CheckpointManager(run=run_name, system_cfg=system_cfg)


def create_minimal_training_setup(
    data_dir: Path,
) -> tuple[TrainerConfig, TrainingEnvironmentConfig, FastConfig, SystemConfig]:
    """Create the minimal training configuration used in fast checkpoint tests."""
    curriculum = env_curriculum(make_arena(num_agents=1))

    trainer_cfg = TrainerConfig(
        total_timesteps=16,
        batch_size=32,
        minibatch_size=16,
        bptt_horizon=4,
        update_epochs=1,
    )

    training_env_cfg = TrainingEnvironmentConfig(
        curriculum=curriculum,
        num_workers=1,
        async_factor=1,
        forward_pass_minibatch_target_size=4,
        vectorization="serial",
        seed=42,
    )

    policy_cfg = FastConfig()
    system_cfg = SystemConfig(
        device="cpu",
        vectorization="serial",
        data_dir=data_dir,
        seed=42,
        local_only=True,
    )

    return trainer_cfg, training_env_cfg, policy_cfg, system_cfg
