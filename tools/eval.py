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

    def __init__(self, cfg: DictConfig, metrics = ["mann_whitney_u", "elo", "glicko2", "time_to_targets"]):
        """
        Store config for later use.
        """
        self.cfg = cfg
        self.metrics = metrics

    def log_metrics(self, stats):
        """
        Logs the various metrics using the stats gathered.
        """
        _, mean_altar_use = get_test_results(MannWhitneyUTest(stats, self.cfg.evaluator.stat_categories['action.use.altar'], mode = 'mean')
            )
        logger.info("\n" + mean_altar_use)

        if "time_to_targets" in self.metrics:
            _, time_to_targets = get_test_results(
                MannWhitneyUTest(stats, self.cfg.evaluator.stat_categories['time_to'], mode = 'mean')
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

    def run_eval(self):
        """
        Main logic of the script: sets up environment, runs evaluation, logs metrics.
        """
        # 1) Initialize environment

        # 2) Start WandB context (if configured)
        with WandbContext(self.cfg) as wandb_run:
            # 3) Get policies
            policy_store = PolicyStore(self.cfg, wandb_run)
            policy = policy_store.policy(self.cfg.evaluator.policy)

            # Possibly load baseline policies
            baselines = []
            if self.cfg.evaluator.baselines.uri:
                baselines = policy_store.policies(self.cfg.evaluator.baselines)

            # 4) Instantiate evaluator & evaluate
            evaluator = hydra.utils.instantiate(self.cfg.evaluator, self.cfg, policy, baselines)
            stats = evaluator.evaluate()
            evaluator.close()

            # 5) Log results
            self.log_metrics(stats)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    driver = Eval(cfg)
    setup_metta_environment(cfg)
    driver.run_eval()


if __name__ == "__main__":
    main()
