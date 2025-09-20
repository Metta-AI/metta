import pufferlib.ocean
import pufferlib.pytorch
import pufferlib.vector
import torch
from pufferlib import pufferl

from mettagrid import MettaGridEnv


class PufferlibPolicy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(env.single_observation_space.shape[0], 128)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(128, 128)),
        )
        self.action_head = torch.nn.Linear(128, env.single_action_space.n)
        self.value_head = torch.nn.Linear(128, 1)

    def forward_eval(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


def env_creator(env: MettaGridEnv):
    return env


def train_pufferlib_policy(env: MettaGridEnv, num_steps: int):
    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=2,
        num_workers=2,
        batch_size=1,
        backend=pufferlib.vector.Multiprocessing,
        env_kwargs={"num_envs": 4096},
    )
    policy = PufferlibPolicy(vecenv.driver_env).cuda()
    args = pufferl.load_config("default")
    args["train"]["env"] = "cogames.cogs_vs_clips"

    trainer = pufferl.PuffeRL(args["train"], vecenv, policy)

    for _ in range(10):
        trainer.evaluate()
        _ = trainer.train()

    trainer.print_dashboard()
    trainer.close()
