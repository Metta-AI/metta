# Generate a graphical trace of multiple runs.

import platform
import webbrowser

import hydra

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite
from metta.util.config import Config, setup_metta_environment
from metta.util.file import http_url
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


# TODO: This job can be replaced with sim now that Simulations create replays
class ReplayJob(Config):
    sim: SimulationSuiteConfig
    policy_uri: str
    selector_type: str
    metric: str
    replay_dir: str = "s3://softmax-public/replays/local"

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

        replay_dir = f"{replay_job.replay_dir}/{cfg.run}"
        if cfg.trainer.get("replay_dry_run", False):
            replay_dir = "/tmp"

        sim_suite = SimulationSuite(replay_job.sim, policy_record, policy_store, replay_dir=replay_dir)
        result = sim_suite.simulate()
        # Only on macos open a browser to the replay
        if platform.system() == "Darwin" and replay_dir is not None:
            replay_url = result.stats_db.get_replay_urls(
                policy_key=policy_record.key(), policy_version=policy_record.version()
            )[0]
            webbrowser.open(f"https://metta-ai.github.io/metta/?replayUrl={http_url(replay_url)}")


if __name__ == "__main__":
    main()
