# Deals with full and partial replays.

import json
import time

import hydra
import torch

from metta.agent.metta_agent import MettaAgent
from metta.agent.policy_store import PolicyStore
from metta.common.util.logging import setup_mettagrid_logger
from metta.common.util.runtime_configuration import setup_mettagrid_environment
from metta.common.util.wandb.wandb_context import WandbContext
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SingleEnvSimulationConfig


class FakeAgent(MettaAgent):
    """
    A fake agent that does nothing, used to run play without requiring a policy to be trained
    """

    def __init__(self):
        pass

    def activate_actions(self, *args):
        pass

    def __call__(self, obs, state):
        num_agents = obs.shape[0]
        return (torch.zeros((num_agents, 2)), None, None, None, None)


class FakePolicyRecord:
    """
    A fake policy record used to return a fake agent.
    """

    def __init__(self):
        self.fake_agent = FakeAgent()
        self.policy_id = "fake"

    def policy_as_metta_agent(self):
        return self.fake_agent

    def policy(self, *args):
        return self.fake_agent


def create_simulation(cfg):
    setup_mettagrid_environment(cfg)

    logger = setup_mettagrid_logger("replay")
    logger.info(f"Replaying {cfg.run}")

    with WandbContext(cfg.wandb, cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)
        if cfg.replay_job.policy_uri is not None:
            policy_record = policy_store.policy(cfg.replay_job.policy_uri)
        else:
            # Set the policy_uri to None to run play without a policy.
            policy_record = FakePolicyRecord()
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
