# Generate a graphical trace of multiple runs.

import time

import hydra

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.util.config import Config, setup_metta_environment
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


def create_simulation(cfg):
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
            replay_dir = None

        print("Create simulation")
        sim = Simulation(
            name="replay",
            config=replay_job.sim,
            policy_pr=policy_record,
            policy_store=policy_store,
            replay_dir=replay_dir,
        )
    return sim


def generate_replay(sim):
    # print("sim._vecenv.envs", sim._vecenv.envs)
    assert len(sim._vecenv.envs) == 1
    # sim._vecenv.envs[0]._replay_writer = sim._replay_writer
    # print("Simulate")
    start = time.time()
    sim.simulate()
    end = time.time()
    print("Simulate time", end - start)
    # print("sim._replay_writer", sim._replay_writer)
    # print("sim._replay_writer.episodes", sim._replay_writer.episodes)
    # assert len(sim._replay_writer.episodes) == 1
    for episode_id, episode_replay in sim._replay_writer.episodes.items():
        # print("episode_id", episode_id)
        # print("episode_replay", episode_replay)
        # print("len(episode_replay.get_replay_data())", len(episode_replay.get_replay_data()))
        return episode_replay.get_replay_data()


if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../../configs", config_name="replay_job")
    def main(cfg):
        start = time.time()
        sim = create_simulation(cfg)
        end = time.time()
        print("Create simulation time", end - start)

        start = time.time()
        replay = generate_replay(sim)
        end = time.time()
        print("len(replay)", len(replay))
        print("Generate replay time", end - start)

    main()
