import os
import signal  # Aggressively exit on ctrl+c
import json # For loading historical elo or glicko scores from local disk
import hydra
from omegaconf import OmegaConf
from rich import traceback
from util.eval_analyzer import Analysis
from rl.wandb.wandb_context import WandbContext
from util.seeding import seed_everything
from agent.policy_store import PolicyStore
from util.stats_library import Glicko2Test
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    traceback.install(show_locals=False)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed, cfg.torch_deterministic)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        evaluator = hydra.utils.instantiate(cfg.evaluator, cfg, policy_store)
        stats = evaluator.evaluate()
        evaluator.close()

        print_significance(cfg, stats)
        print_elo(cfg, stats)
        print_glicko(cfg, stats)

def _load_scores(scores_path):
    if scores_path and os.path.exists(scores_path):
        with open(scores_path, "r") as file:
            return json.load(file)
    return {}

def _save_scores(scores_path, scores):
    if scores_path:
        with open(scores_path, "w") as file:
            json.dump(scores, file, indent=4)

def print_elo(cfg, stats):
    old_scores = _load_scores(cfg.evaluator.baselines.elo_scores_path)

    elo_analysis = Analysis(stats, eval_method='elo_1v1', stat_category='altar',
                            initial_elo_scores=old_scores)
    print(elo_analysis.get_display_results())

    _save_scores(cfg.evaluator.baselines.elo_scores_path,
                 elo_analysis.get_updated_historicals())

def print_glicko(cfg, stats):
    old_scores = _load_scores(cfg.evaluator.baselines.glicko_scores_path)

    test = Glicko2Test(stats, 'action.use.altar', old_scores)

    print(test.get_display_results())

    _save_scores(
        cfg.evaluator.baselines.glicko_scores_path,
        test.get_updated_historicals())


def print_significance(cfg, stats):
    p_analysis = Analysis(stats, eval_method='1v1', stat_category='all')
    print(p_analysis.get_display_results())

if __name__ == "__main__":
    main()
