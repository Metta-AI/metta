import os
import signal  # Aggressively exit on ctrl+c

import hydra
from omegaconf import OmegaConf
from rich import traceback
from rl.pufferlib.evaluator import PufferEvaluator
from rl.pufferlib.policy import load_policy_from_uri
# from util.stats import print_policy_stats
from util.eval_analyzer import Analysis
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):

    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    with WandbContext(cfg) as wandb_run:
        policy = load_policy_from_uri(cfg.evaluator.policy_uri, cfg, wandb_run)
        baselines = []
        for uri in cfg.evaluator.baseline_uris:
            baselines.append(load_policy_from_uri(uri, cfg, wandb_run))
        baselines = baselines[0:cfg.evaluator.max_baselines]
        evaluator = hydra.utils.instantiate(cfg.evaluator, cfg, policy, baselines)
        stats = evaluator.evaluate()

        p_analysis = Analysis(stats, eval_method='1v1', stat_category='all')

        # Get raw results
        p_results = p_analysis.get_results()
        print(p_results) # Printing raw results for demonstration purposes

        # Print formatted results
        print(p_analysis.get_display_results())

        #delete below after testing. Some dummy historical elo scores :)
        initial_elo_scores = {
            'b.gpop.simple.2:v0': {'score': 500, 'episodes': 37},
            'Policy 2': {'score': 1500, 'episodes': 12}
        }

        elo_analysis = Analysis(stats, eval_method='elo_1v1', stat_category='altar', initial_elo_scores=initial_elo_scores)

        # Get raw results
        elo_results = elo_analysis.get_results()
        print(elo_results) # Printing raw results for demonstration purposes

        # Print formatted results
        print(elo_analysis.get_display_results())


if __name__ == "__main__":
    main()
