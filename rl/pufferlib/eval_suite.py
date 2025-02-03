# suite/eval_suite.py

import logging
import copy
import hydra
from omegaconf import DictConfig, OmegaConf

# Import your evaluator class
from rl.pufferlib.evaluator import PufferEvaluator
from rl.pufferlib.policy_record import PolicyRecord

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="suite_config")
def main(cfg: DictConfig):
    logger.info("===== Beginning Evaluation Suite =====")

    logger.info(f"Global Policy URI: {cfg.policy_uri}")
    logger.info(f"Global Baseline URI: {cfg.baseline_uri}")

    # We iterate over each evaluation item in the list
    for eval_item in cfg.evals:
        env_name = eval_item.env  # e.g. "multi_room_cylinder"
        metrics_to_log = eval_item.get("metrics_to_log", [])

        # 1) Create a working copy of the base config
        #    (which currently includes the base evaluator config).
        eval_cfg = copy.deepcopy(cfg)

        # 2) Merge environment config into our copy
        env_override = hydra.compose(config_name=f"{env_name}.yaml")
        # Now we merge env_override into eval_cfg:
        eval_cfg = OmegaConf.merge(eval_cfg, env_override)

        eval_cfg.policy.uri = cfg.policy_uri
        eval_cfg.baselines.uri = cfg.baseline_uri

        logger.info(f"--- Running evaluation for environment: {env_name} ---")
        logger.info(f"Metrics to log: {metrics_to_log}")
        logger.info(f"Merged config:\n{OmegaConf.to_yaml(eval_cfg, resolve=True)}")

        policy_record = PolicyRecord(eval_cfg.policy.uri)
        baseline_records = [PolicyRecord(uri) for uri in eval_cfg.baselines.uri]

        evaluator = PufferEvaluator(eval_cfg, policy_record, baseline_records)
        evaluator.evaluate()

        evaluator.close()

    logger.info("===== All Evaluations Complete =====")


if __name__ == "__main__":
    main()
