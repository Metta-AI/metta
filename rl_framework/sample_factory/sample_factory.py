import json
import hydra

from numpy import mean, rec
from omegaconf import OmegaConf
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.enjoy import enjoy
from rl_framework.rl_framework import EvaluationResult, RLFramework
from rl_framework.sample_factory.sample_factory_agent_wrapper import SampleFactoryAgentWrapper
from rl_framework.sample_factory.sample_factory_env_wrapper import SampleFactoryEnvWrapper


def make_env_func(full_env_name, sf_cfg, sf_env_config, render_mode):
    env_cfg = OmegaConf.create(json.loads(sf_cfg.env_cfg))
    env = hydra.utils.instantiate(env_cfg, render_mode=render_mode)
    env = SampleFactoryEnvWrapper(env, env_id=0)
    return env

def make_agent_func(sf_cfg, obs_space, action_space):
    env_cfg = OmegaConf.create(json.loads(sf_cfg.env_cfg))
    env = hydra.utils.instantiate(env_cfg, render_mode="human")

    agent_cfg = OmegaConf.create(json.loads(sf_cfg.agent_cfg))
    agent_cfg.observation_encoders.grid_obs.feature_names = env.grid_features
    agent_cfg.observation_encoders.global_vars.feature_names = env.global_features
    agent = hydra.utils.instantiate(agent_cfg, obs_space, action_space, _recursive_=False)
    return SampleFactoryAgentWrapper(agent)

class SampleFactoryFramework(RLFramework):
    def __init__(self, cfg, **sf_args):
        super().__init__(OmegaConf.create(cfg))
        self.sf_args = [
            f"--{k}={v}" for k, v in cfg.framework.sample_factory.items()
        ] + [
            f"--{k}={v}" for k, v in cfg.agent.core.items() if k.startswith("rnn_")
        ]
        if cfg.wandb.track:
            self.sf_args.append("--with_wandb=True")
            self.sf_args.append(f"--wandb_user={cfg.wandb.entity}")
            self.sf_args.append(f"--wandb_project={cfg.wandb.project}")
            self.sf_args.append(f"--wandb_group={cfg.wandb.group}")

        register_env(cfg.env.name, make_env_func)
        self.sf_args.append(f"--env={cfg.env.name}")
        self.sf_args.append(
            "--env_cfg=" +
            json.dumps(OmegaConf.to_container(cfg.env, resolve=True)))
        self.sf_args.append(
            "--agent_cfg=" +
            json.dumps(OmegaConf.to_container(cfg.agent, resolve=True)))

        model_factory = global_model_factory()
        model_factory.register_actor_critic_factory(make_agent_func)

    def setup(self, evaluation=False):
        print("SampleFactory Args: ", self.sf_args)
        sf_parser, cfg = parse_sf_args(self.sf_args, evaluation=evaluation)
        sf_parser.add_argument("--env_cfg", type=str, default=None)
        sf_parser.add_argument("--agent_cfg", type=str, default=None)
        sf_cfg = parse_full_cfg(sf_parser, self.sf_args)
        return sf_cfg

    def train(self):
        sf_cfg = self.setup()
        status =  run_rl(sf_cfg)
        return status

    def evaluate(self):
        sf_cfg = self.setup(evaluation=True)
        sf_cfg.max_num_frames = self.cfg.eval.max_steps
        sf_cfg.save_video = False
        sf_cfg.eval_env_frameskip = 1

        status = enjoy(sf_cfg, render_mode="rgb_array")

        return EvaluationResult(
            reward=status["reward"],
            frames=status["frames"],
        )
