# Generate a graphical trace of multiple runs.

import platform
import webbrowser

import hydra

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import SimulationSuite
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import Config, setup_metta_environment
from metta.util.file import s3_url
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


class ReplayJob(Config):
    sim: SimulationSuiteConfig
    policy_uri: str
    selector_type: str
    metric: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg):
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("replay")
    logger.info(f"Replaying {cfg.run}")

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        replay_job = ReplayJob(cfg.replay_job)
        policy_record = policy_store.policy(replay_job.policy_uri)

        for name, sim in replay_job.sim.simulations.items():
            sim.replay_path = f"s3://softmax-public/replays/local/{cfg.run}/{name}/replay.json"
        sim_suite = SimulationSuite(replay_job.sim, policy_record, policy_store, wandb_run=wandb_run)
        sim_suite.simulate(dry_run=cfg.trainer.get("replay_dry_run", False))
        # Only on macos open a browser to the replay
        # TODO: This wont be quite the right URL if num_episodes >1  num_envs > 1
        # see Simulation._get_replay_path()
        first_sim_path = list(replay_job.sim.simulations.values())[0].replay_path
        if platform.system() == "Darwin":
            webbrowser.open(f"https://metta-ai.github.io/metta/?replayUrl={s3_url(first_sim_path)}")


if __name__ == "__main__":
    main()
