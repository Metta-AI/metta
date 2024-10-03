import os
import signal  # Aggressively exit on ctrl+c

import hydra
from omegaconf import OmegaConf
from rich import traceback
from rl.pufferlib.evaluator import PufferEvaluator
from rl.pufferlib.policy import load_policy_from_uri
# from util.stats import print_policy_stats
from util.eval_analyzer import print_policy_stats
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything
from rl.pufferlib.policy import load_policies_from_dir
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    with WandbContext(cfg) as wandb_run:
        policy = load_policy_from_uri(cfg.eval.policy_uri, cfg, wandb_run)
        baselines = []
        for uri in cfg.eval.baseline_uris:
            baselines.append(load_policy_from_uri(uri, cfg, wandb_run))
        baselines = baselines[0:cfg.eval.max_baselines]
        evaluator = PufferEvaluator(cfg, policy, baselines)
        stats = evaluator.evaluate()
        # print_policy_stats(stats)
        print_policy_stats(stats, '1v1', 'all')
        print_policy_stats(stats, 'elo_1v1', 'altar')


if __name__ == "__main__":
    main()
