# Deals with full and partial replays.

import json
import logging
import time

from omegaconf import DictConfig

from metta.agent.mocks import MockPolicyRecord
from metta.common.wandb.wandb_context import WandbContext
from metta.rl.env_config import create_env_config
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig
from metta.util.metta_script import metta_script
from tools.utils import get_policy_store_from_cfg

logger = logging.getLogger(__name__)


def create_simulation(cfg):
    logger.info(f"Replaying {cfg.run}")

    # Create env config - use default if no specific env provided
    if hasattr(cfg.replay_job, 'env') and cfg.replay_job.env:
        # If env path is provided, create from that
        from omegaconf import OmegaConf
        env_cfg_dict = OmegaConf.load(cfg.replay_job.env)
        env_cfg = create_env_config(env_cfg_dict) 
    else:
        # Use default env config for basic play session
        from metta.rl.env_config import EnvConfig
        env_cfg = EnvConfig(device=cfg.replay_job.device)

    with WandbContext(cfg.wandb, cfg) as wandb_run:
        policy_store = get_policy_store_from_cfg(cfg, wandb_run)
        if cfg.replay_job.policy_uri is not None:
            policy_record = policy_store.policy_record(cfg.replay_job.policy_uri)
        else:
            # Set the policy_uri to None to run play without a policy.
            policy_record = MockPolicyRecord(run_name="replay_run", uri=None)
        sim_config = SingleEnvSimulationConfig(cfg.replay_job.sim)

        sim_name = sim_config.env.split("/")[-1]
        replay_dir = f"{cfg.replay_job.replay_dir}/{cfg.run}"

        sim = Simulation(
            sim_name,
            sim_config,
            policy_record,
            policy_store,
            device=cfg.device,
            vectorization=env_cfg.vectorization,
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
