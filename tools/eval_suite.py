# eval_suite.py

import logging
import hydra
from hydra import compose
from omegaconf import DictConfig
from mettagrid.config.config import setup_metta_environment
from tools.eval import Eval

logger = logging.getLogger("eval_suite.py")

class EvalSuite:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = cfg.eval_suite.policy_uri
        self.baselines = cfg.eval_suite.baseline_uri

        self.evals = cfg.eval_suite.evals

    def run_all(self):

        all_baseline_stats = [] #across all evals
        all_policy_stats = [] #across all evals
        for eval in self.evals:
            current_cfg = self.cfg.copy()

            map_name = eval.env

            logger.info(f"Running evaluation for environment: {map_name} with npcs: {eval.npc_policy_uri}"+ "\n")
            logger.info(f"Baseline policies: {self.baselines}"+ "\n")
            logger.info(f"Candidate policy: {self.policy}"+ "\n")

            env_cfg = compose(config_name=map_name)

            current_cfg.env = env_cfg.env.mettagrid

            # what we are calling baselines for the evaluator is the npcs (the opponents)
            #If we have no baselines it makes the policy the baseline / NPC
            current_cfg["evaluator"]["baselines"]["uri"] = eval.npc_policy_uri

            #first run the baseline policies against the npcs for this eval
            baseline_stats = []
            for baseline in self.baselines:
                current_cfg["evaluator"]["policy"]["uri"] = baseline #the "candidate policy" here is baseline 1
                driver = Eval(current_cfg, metrics = eval.metrics_to_log, env_name = map_name)
                stats = driver.run_eval(log = False)
                baseline_stats.append(stats)

                print(f"Collected stats for baseline {baseline} against npc: {eval.npc_policy_uri}" + "\n")
                driver.log_metrics(stats)
                print("--------------------------------")
            all_baseline_stats.append(baseline_stats)

            #then run the policy against the npcs
            current_cfg["evaluator"]["policy"]["uri"] = self.policy

            # 5) Instantiate your evaluator class (Eval), then run
            # Temporarily disable logging

            driver = Eval(current_cfg, metrics = eval.metrics_to_log, env_name = map_name)
            policy_stats = driver.run_eval(log = False)
            all_policy_stats.append(policy_stats)

            print(f"Collected stats for candidate {self.policy} against npc: {eval.npc_policy_uri}" + "\n")
            driver.log_metrics(policy_stats)

            #TODO rather than logging, somehow aggregate the stats for policies and baselines and then log them

        logger.info("=== Eval Suite Complete ===")
        #now its time to do some fancy analysis to compare policy performance against baseline performance (average baseline performance) across each eval


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    driver = EvalSuite(cfg)
    setup_metta_environment(cfg)
    driver.run_all()

if __name__ == "__main__":
    main()
