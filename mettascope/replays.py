# Deals with full and partial replays.

import json
import logging
import time

from omegaconf import DictConfig

from metta.agent.mocks import MockPolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbContext
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig
from metta.util.metta_script import metta_script

logger = logging.getLogger(__name__)


def create_simulation(cfg):
    logger.info(f"Replaying {cfg.run}")

    with WandbContext(cfg.wandb, cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        if cfg.replay_job.policy_uri is not None:
            policy_record = policy_store.policy_record(cfg.replay_job.policy_uri)
        else:
            # Set the policy_uri to None to run play without a policy.
            policy_record = MockPolicyRecord(policy_store=None, run_name="replay_run", uri=None)
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


def generate_replay(sim: Simulation) -> dict:
    assert len(sim._vecenv.envs) == 1, "Replay generation requires a single environment"
    start = time.time()
    sim.simulate()
    end = time.time()
    print("Simulate time", end - start)
    assert len(sim._replay_writer.episodes) == 1, "Expected exactly one replay episode"
    for _, episode_replay in sim._replay_writer.episodes.items():
        return episode_replay.get_replay_data()
    return {}


def main(cfg: DictConfig) -> None:
    start = time.time()
    sim = create_simulation(cfg)
    end = time.time()
    print("Create simulation time", end - start)
    start = time.time()
    replay = generate_replay(sim)
    end = time.time()
    print("replay: ", len(json.dumps(replay)), "bytes")
    print("Generate replay time", end - start)


metta_script(main, "replay_job")
