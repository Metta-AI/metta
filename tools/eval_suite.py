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
        self.cfg["evaluator"]["policy"]["uri"] = cfg.eval_suite.policy_uri
        #this should be a list of uris
        self.cfg["evaluator"]["baselines"]["uri"] = cfg.eval_suite.baseline_uri 
        self.evals = cfg.eval_suite.evals
    
    def run_all(self):
        for eval in self.evals:
            current_cfg = self.cfg.copy()

            env_name = eval["env"]  # e.g. 'env/mettagrid/cylinder'

            logger.info(f"\n--- Running evaluation for environment: {env_name} ---")

            env_cfg = compose(config_name=env_name)

            # 2) Merge environment config into a fresh copy of the suite config
            current_cfg["env"] = env_cfg.env["mettagrid"]

            # 5) Instantiate your evaluator class (Eval), then run
            driver = Eval(current_cfg, metrics = eval.metrics_to_log, env_name = env_name)
            driver.run_eval()

            logger.info("=== Eval Suite Complete ===")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    driver = EvalSuite(cfg)
    setup_metta_environment(cfg)
    driver.run_all()

if __name__ == "__main__":
    main()
