import logging

import hydra
from omegaconf import DictConfig
from mettagrid.config.config import setup_metta_environment
from agent.policy_store import PolicyStore
from rl.wandb.wandb_context import WandbContext
from util.stats_library import (
    EloTest,
    Glicko2Test,
    MannWhitneyUTest,
    StatisticalTest,
    get_test_results
)

logger = logging.getLogger("eval.py")


class Eval:

    def __init__(self, cfg: DictConfig, metrics = ["mann_whitney_u", "elo", "glicko2", "time_to_targets"], env_name = None):
        """
        Store config for later use.
        """
        self.cfg = cfg
        self.metrics = metrics
        self.env_name = env_name

    def log_metrics(self, stats):
        """
        Logs the various metrics using the stats gathered.
        """
        logger.setLevel(logging.INFO)

        _, mean_altar_use = get_test_results(MannWhitneyUTest(stats, self.cfg.evaluator.stat_categories['action.use.altar'], mode = 'mean', label = self.env_name)
            )
        logger.info("\n" + mean_altar_use)

        if "time_to_targets" in self.metrics:
            _, time_to_targets = get_test_results(
                MannWhitneyUTest(stats, self.cfg.evaluator.stat_categories['time_to'], mode = 'mean', label = self.env_name)
            )
            logger.info("\n" + time_to_targets)


        if "mann_whitney_u" in self.metrics:
            # Generic Mann-Whitney on "all" stats
            _, fr = get_test_results(
                MannWhitneyUTest(stats, self.cfg.evaluator.stat_categories['all'])
            )
            logger.info("\n" + fr)

        if "elo" in self.metrics:
            # Elo test on "altar" category
            _, fr = get_test_results(
                EloTest(stats, self.cfg.evaluator.stat_categories['altar']),
                self.cfg.evaluator.baselines.elo_scores_path
            )
            logger.info("\n" + fr)

        if "glicko2" in self.metrics:
            # Glicko2 test on "altar" category
            _, fr = get_test_results(
                Glicko2Test(stats, self.cfg.evaluator.stat_categories['altar']),
                self.cfg.evaluator.baselines.glicko_scores_path
            )
            logger.info("\n" + fr)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_metta_environment(cfg)

    with WandbContext(cfg) as wandb_run:
        policy_store = PolicyStore(cfg, wandb_run)

        eval = hydra.utils.instantiate(
            cfg.eval,
            policy_store,
            cfg.env,
            _recursive_=False
        )
        stats = eval.evaluate()
        # log_metrics(stats)

        return stats


if __name__ == "__main__":
    main()
