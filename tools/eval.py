import logging

import hydra
from agent.policy_store import PolicyStore
from mettagrid.config.config import setup_metta_environment
from rl.wandb.wandb_context import WandbContext
from util.stats_library import EloTest, Glicko2Test, MannWhitneyUTest, get_test_results

logger = logging.getLogger("eval.py")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        policy = policy_store.policy(cfg.evaluator.policy)
        baselines = []
        if cfg.evaluator.baselines.uri:
            baselines = policy_store.policies(cfg.evaluator.baselines)
        evaluator = hydra.utils.instantiate(cfg.evaluator, cfg, policy, baselines)
        stats = evaluator.evaluate()
        evaluator.close()

        _, fr = get_test_results(MannWhitneyUTest(stats, cfg.evaluator.stat_categories['all']))
        logger.info("\n" + fr)
        _, fr = get_test_results(EloTest(stats, cfg.evaluator.stat_categories['altar']), cfg.evaluator.baselines.elo_scores_path)
        logger.info("\n" + fr)
        _, fr = get_test_results(Glicko2Test(stats, cfg.evaluator.stat_categories['altar']), cfg.evaluator.baselines.glicko_scores_path)
        logger.info("\n" + fr)

if __name__ == "__main__":
    main()
