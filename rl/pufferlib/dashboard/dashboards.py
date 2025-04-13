
from omegaconf import OmegaConf

from rl.carbs.carbs_controller import CarbsController
from rl.pufferlib.dashboard.carbs import Carbs
from rl.pufferlib.dashboard.dashboard import Dashboard
from rl.pufferlib.dashboard.logs import Logs
from rl.pufferlib.dashboard.policy import Policy
from rl.pufferlib.dashboard.training import Training
from rl.pufferlib.dashboard.utilization import Utilization
from rl.pufferlib.dashboard.wandb import WanDb
from rl.pufferlib.train import PufferTrainer


def train_dashboard(trainer: PufferTrainer):
    return Dashboard(trainer.cfg, components=[
        Utilization(),
        WanDb(trainer.cfg.wandb),
        Training(trainer),
        Policy(trainer.policy_checkpoint),
        Logs(logs_path),
    ])

def eval_dashboard(cfg: OmegaConf, stats: Dict):
    return Dashboard(cfg, components=[
        Utilization(),
        WanDb(cfg.wandb),
        Eval(stats),
    ])

def sweep_dashboard(cfg: OmegaConf, carbs_controller: CarbsController):
    return Dashboard(cfg, components=[
        Utilization(),
        WanDb(cfg.wandb),
        Carbs(carbs_controller),
        Logs(logs_path),
    ])

