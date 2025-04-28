# Generate a graphical trace of multiple runs.

import platform
import webbrowser

import hydra

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.util.config import Config, setup_metta_environment
from metta.util.file import s3_url
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


class ReplayJob(Config):
    sim: SimulationConfig
    policy_uri: str


@hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
def main(cfg):
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        replay_job = ReplayJob(cfg.replay_job)
        policy_record = policy_store.policy(replay_job.policy_uri)
        replay_job.sim.replay_path = f"s3://softmax-public/replays/local/{cfg.run}/replay.json.z"
        sim = Simulation(replay_job.sim, policy_record, policy_store, wandb_run=wandb_run)
        sim.simulate()
        # Only on macos open a browser to the replay
        if platform.system() == "Darwin":
            webbrowser.open(f"https://metta-ai.github.io/metta/?replayUrl={s3_url(replay_job.sim.replay_path)}")


if __name__ == "__main__":
    main()
