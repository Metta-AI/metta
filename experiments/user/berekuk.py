import yaml
from metta.cogworks.curriculum import env_curriculum
from metta.mettagrid.config.envs import make_arena
from metta.rl.trainer_config import TrainerConfig
from metta.tools.train import TrainTool


def train(run: str) -> TrainTool:
    return TrainTool(
        run=run,
        trainer=TrainerConfig(
            batch_size=1024,
            minibatch_size=1024,
            forward_pass_minibatch_target_size=2,
            curriculum=env_curriculum(make_arena(num_agents=4)),
        ),
        policy_architecture=yaml.safe_load(open("configs/agent/fast.yaml")),
    )
