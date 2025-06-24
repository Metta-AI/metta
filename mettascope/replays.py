# Deals with full and partial replays.

import json
import time

import hydra

from metta.agent.policy_store import PolicyStore
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig
from metta.util.config import setup_metta_environment
from metta.util.logging import setup_mettagrid_logger
from metta.util.runtime_configuration import setup_mettagrid_environment
from metta.util.wandb.wandb_context import WandbContext


def create_simulation(cfg):
    setup_metta_environment(cfg)
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("replay")
    logger.info(f"Replaying {cfg.run}")

    with WandbContext(cfg.wandb, cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        policy_record = policy_store.policy(cfg.replay_job.policy_uri)
        sim_config = SingleEnvSimulationConfig(cfg.replay_job.sim)

        sim_name = sim_config.env.split("/")[-1]
        replay_dir = f"{cfg.replay_job.replay_dir}/{cfg.run}"

        sim = Simulation(
            sim_name,
            sim_config,
            policy_record,
            policy_store,
            device=cfg.device,
            vectorization=cfg.vectorization,
            stats_dir=cfg.replay_job.stats_dir,
            replay_dir=replay_dir,
        )
    return sim


def generate_replay(sim) -> dict:
    assert len(sim._vecenv.envs) == 1, "Replay generation requires a single environment"
    start = time.time()
    sim.simulate()
    end = time.time()
    print("Simulate time", end - start)
    assert len(sim._replay_writer.episodes) == 1, "Expected exactly one replay episode"
    for _, episode_replay in sim._replay_writer.episodes.items():
        return episode_replay.get_replay_data()
    return {}


if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../configs", config_name="replay_job")
    def main(cfg):
        start = time.time()
        sim = create_simulation(cfg)
        end = time.time()
        print("Create simulation time", end - start)
        start = time.time()
        replay = generate_replay(sim)
        end = time.time()
        print("replay: ", len(json.dumps(replay)), "bytes")
        print("Generate replay time", end - start)

    main()
