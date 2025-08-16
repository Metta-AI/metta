import yaml
from experiments.evals.navigation import make_navigation_eval
from metta.cogworks.curriculum import env_curriculum
from metta.mettagrid.config.envs import make_arena
from metta.rl.trainer_config import TrainerConfig
from metta.tools.sim import SimTool
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


def sim_navigation(policy_uri: str) -> SimTool:
    return SimTool(
        simulations=make_navigation_eval(),
        policy_uris=[policy_uri],
        stats_dir="/tmp/stats",
        replay_dir="./train_dir/replays",
        stats_db_uri="wandb://stats/navigation_db",
        stats_server_uri="http://localhost:8000",
    )
